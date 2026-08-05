"""Execution wrapper for the opt-in tiered RAG autotuning policy (§2, §4).

This module runs the fixed runtime order — kill switch, exact-cache probe,
workload description, frozen retrieval, the pure decision, then execution behind
the typed fallback boundary — while recording per-phase timings, and returns an
:class:`~helion.autotuner.rag.types.ExecutionOutcome`.

It is a pure orchestration layer: every side-effecting dependency (the exact-cache
probe, workload description, retriever, decision runner, baseline tuner, clock,
and deadline check) is injected, so the whole wrapper is exercised without CUDA,
faiss, or any provider. The only failure boundary is
:func:`~helion.autotuner.rag.fallback.execute_with_fallback`; no other defensive
handling is added here.

See ``docs/rag_autotuning_experiment.md``.
"""

from __future__ import annotations

import time
from typing import Callable
from typing import Protocol

from .fallback import classify_fallback
from .policy import decide
from .types import REASON_RAG_DISABLED
from .types import BaselineSearch
from .types import Decision
from .types import ExactHit
from .types import ExactProbeResult
from .types import ExactReadError
from .types import ExecutionOutcome
from .types import FallbackReason
from .types import Phase
from .types import PhaseTimingEvent
from .types import RetrievalEvidence
from .types import TreatmentConfig
from .types import WorkloadDescriptor

# Stable ``BaselineSearch.reason`` strings owned by the execution wrapper. These
# are runtime-order outcomes (an existing exact hit, a fail-closed read error),
# distinct from the pure-policy reasons in :mod:`helion.autotuner.rag.types`.
REASON_EXISTING_EXACT_HIT = "existing_exact_hit"
REASON_EXACT_READ_ERROR = "exact_read_error"
REASON_RETRIEVAL_FAILURE = "retrieval_failure"


class DecisionRunner(Protocol):
    """Executes one RAG-active :class:`Decision` and returns the selected result.

    The concrete runner performs the generation/seeding/compilation/correctness/
    benchmarking work for ``ExactReuse``/``RetrievalSeededSearch``/
    ``ContextualSearch`` and emits its own inner phase timings; the wrapper only
    sequences it behind the fallback boundary. ``BaselineSearch`` never reaches
    the runner — it is the frozen baseline and runs directly.
    """

    def run(self, decision: Decision) -> object: ...


def execute_rag_attempt(
    treatment: TreatmentConfig,
    *,
    probe: Callable[[], ExactProbeResult],
    describe: Callable[[], WorkloadDescriptor],
    retrieve: Callable[[WorkloadDescriptor], RetrievalEvidence],
    runner: DecisionRunner,
    baseline: Callable[[], object],
    failure_fallback: Callable[[], object] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    deadline_exceeded: Callable[[], bool] | None = None,
    phase_observer: Callable[[PhaseTimingEvent], None] | None = None,
) -> ExecutionOutcome:
    """Run the fixed runtime order and return the terminal execution outcome.

    ``probe`` reads Helion's real exact cache, ``describe`` builds the
    :class:`WorkloadDescriptor` (only on a true miss), ``retrieve`` performs the
    frozen lookup, ``runner`` executes a RAG-active decision, and ``baseline``
    executes a deterministic ``BaselineSearch`` decision without changing tuner
    semantics. ``failure_fallback`` is the frozen recovery tuner used only for a
    typed operational failure; it defaults to ``baseline`` for compatibility.
    ``clock`` and ``deadline_exceeded`` are injected so timing and right-censoring
    are deterministic in tests.
    """
    if failure_fallback is None:
        failure_fallback = baseline
    timings: list[PhaseTimingEvent] = []
    start = clock()

    def _record_phase(phase: Phase, seconds: float) -> None:
        event = PhaseTimingEvent(phase, seconds)
        timings.append(event)
        if phase_observer is not None:
            phase_observer(event)

    def _finish(
        decision: Decision,
        result: object,
        *,
        fallback_reason: FallbackReason | None,
    ) -> ExecutionOutcome:
        # A deadline-terminated arm is a right-censoring boundary, not a validated
        # observation; the actual accumulated time is reported either way (§4).
        censored = bool(deadline_exceeded()) if deadline_exceeded is not None else False
        total = clock() - start
        _record_phase(Phase.TOTAL, total)
        return ExecutionOutcome(
            decision=decision,
            result=result,
            validated=not censored,
            accumulated_seconds=total,
            censored=censored,
            fallback_reason=fallback_reason,
            timings=tuple(timings),
        )

    # Step 1: kill switch. A disabled treatment is a normal baseline state, not a
    # failure, so no fallback reason is attributed.
    if not treatment.rag_enabled:
        result = baseline()
        return _finish(
            BaselineSearch(REASON_RAG_DISABLED), result, fallback_reason=None
        )

    # Steps 2-3: probe Helion's real exact cache.
    lookup_start = clock()
    probe_result = probe()
    _record_phase(Phase.LOOKUP, clock() - lookup_start)

    if isinstance(probe_result, ExactHit):
        # An existing exact hit is never attributed to RAG; no retrieval performed.
        return _finish(
            BaselineSearch(REASON_EXISTING_EXACT_HIT),
            probe_result.config,
            fallback_reason=None,
        )
    if isinstance(probe_result, ExactReadError):
        # Fail closed: a read error routes to baseline and is recorded separately,
        # never reclassified as a miss.
        result = failure_fallback()
        return _finish(
            BaselineSearch(REASON_EXACT_READ_ERROR),
            result,
            fallback_reason=FallbackReason.EXACT_READ_ERROR,
        )

    # Step 4: true miss — build the descriptor and query frozen retrieval.
    embedding_start = clock()
    try:
        evidence = retrieve(describe())
    except Exception as exception:
        retrieval_fallback = classify_fallback(exception)
        if retrieval_fallback is None:
            raise
        _record_phase(Phase.EMBEDDING, clock() - embedding_start)
        fallback_start = clock()
        result = failure_fallback()
        _record_phase(Phase.FALLBACK, clock() - fallback_start)
        return _finish(
            BaselineSearch(REASON_RETRIEVAL_FAILURE),
            result,
            fallback_reason=retrieval_fallback,
        )
    _record_phase(Phase.EMBEDDING, clock() - embedding_start)

    # The pure policy decision is timed separately from embedding/index lookup.
    decision_start = clock()
    decision = decide(evidence, treatment)
    _record_phase(Phase.DECISION, clock() - decision_start)

    # Steps 5-6: a BaselineSearch decision is the frozen baseline itself and runs
    # directly; every RAG-active decision runs behind the typed fallback boundary.
    if isinstance(decision, BaselineSearch):
        result = baseline()
        return _finish(decision, result, fallback_reason=None)

    try:
        result = runner.run(decision)
        fallback_reason = None
    except Exception as exception:
        fallback_reason = classify_fallback(exception)
        if fallback_reason is None:
            raise
        fallback_start = clock()
        result = failure_fallback()
        _record_phase(Phase.FALLBACK, clock() - fallback_start)
    return _finish(decision, result, fallback_reason=fallback_reason)
