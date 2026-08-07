from __future__ import annotations

import pytest

from helion import exc
from helion.autotuner.rag import BaselineSearch
from helion.autotuner.rag import ContextualSearch
from helion.autotuner.rag import ExactHit
from helion.autotuner.rag import ExactMiss
from helion.autotuner.rag import ExactProbeResult
from helion.autotuner.rag import ExactReadError
from helion.autotuner.rag import FallbackReason
from helion.autotuner.rag import LookupTier
from helion.autotuner.rag import Phase
from helion.autotuner.rag import PhaseTimingEvent
from helion.autotuner.rag import RetrievalEvidence
from helion.autotuner.rag import RetrievalSeededSearch
from helion.autotuner.rag import RetrievedNeighbor
from helion.autotuner.rag import TreatmentConfig
from helion.autotuner.rag import TunerMode
from helion.autotuner.rag import WorkloadDescriptor
from helion.autotuner.rag.execution import execute_rag_attempt

_WORKLOAD = WorkloadDescriptor(
    kernel_name="k",
    kernel_source="src",
    input_shapes="[1]",
    dtypes="fp16",
    hardware="h100",
    backend="triton",
    structural_fingerprint_hash="hash",
)
_NEIGHBOR = RetrievedNeighbor(
    kernel_name="k", input_shapes="[1]", dtypes="fp16", score=0.9
)


def _treatment(**kwargs: object) -> TreatmentConfig:
    params: dict[str, object] = {"treatment_id": "t", "rag_enabled": True}
    params.update(kwargs)
    return TreatmentConfig(**params)  # pyrefly: ignore [bad-argument-type]


class _FakeProbe:
    def __init__(self, result: ExactProbeResult) -> None:
        self.calls = 0
        self._result = result

    def __call__(self) -> ExactProbeResult:
        self.calls += 1
        return self._result


class _FakeRetriever:
    def __init__(self, evidence: RetrievalEvidence) -> None:
        self.calls = 0
        self._evidence = evidence

    def __call__(self, descriptor: WorkloadDescriptor) -> RetrievalEvidence:
        self.calls += 1
        return self._evidence


class _FakeBaseline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "baseline"


class _FakeRunner:
    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[object] = []
        self._raises = raises

    def run(self, decision: object) -> str:
        self.calls.append(decision)
        if self._raises is not None:
            raise self._raises
        return "ran"


class _FakeClock:
    """Deterministic monotonic clock advancing a fixed step on every read."""

    def __init__(self, step: float = 1.0) -> None:
        self.now = 0.0
        self._step = step

    def __call__(self) -> float:
        self.now += self._step
        return self.now


def _semantic(*, neighbors: tuple[RetrievedNeighbor, ...] = (_NEIGHBOR,)):
    return RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=neighbors)


def test_kill_switch_off_runs_baseline_without_probe_or_retrieval():
    probe = _FakeProbe(ExactMiss())
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(rag_enabled=False),
        probe=probe,
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, BaselineSearch)
    assert outcome.decision.reason == "rag_disabled"
    assert outcome.fallback_reason is None
    assert baseline.calls == 1
    assert probe.calls == 0
    assert retriever.calls == 0
    assert runner.calls == []


def test_existing_exact_hit_not_attributed_to_rag():
    cached = object()
    probe = _FakeProbe(ExactHit(config=cached))
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(),
        probe=probe,
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, BaselineSearch)
    assert outcome.decision.reason == "existing_exact_hit"
    assert outcome.validated is True
    assert outcome.fallback_reason is None
    assert outcome.result is cached
    assert retriever.calls == 0
    assert runner.calls == []
    assert baseline.calls == 0


def test_exact_read_error_fails_closed_to_baseline():
    probe = _FakeProbe(ExactReadError(error="boom"))
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(),
        probe=probe,
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, BaselineSearch)
    assert outcome.fallback_reason == FallbackReason.EXACT_READ_ERROR
    assert baseline.calls == 1
    assert retriever.calls == 0
    assert runner.calls == []


def test_tier1_lfbo_runs_retrieval_seeded_search():
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, RetrievalSeededSearch)
    assert outcome.result == "ran"
    assert outcome.fallback_reason is None
    assert retriever.calls == 1
    assert len(runner.calls) == 1
    assert isinstance(runner.calls[0], RetrievalSeededSearch)
    assert baseline.calls == 0


def test_tier1_llm_runs_contextual_search():
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LLM),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, ContextualSearch)
    assert len(runner.calls) == 1
    assert isinstance(runner.calls[0], ContextualSearch)
    assert baseline.calls == 0


def test_tier2_miss_delegates_to_baseline():
    retriever = _FakeRetriever(RetrievalEvidence(tier=LookupTier.MISS))
    baseline = _FakeBaseline()
    runner = _FakeRunner()

    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, BaselineSearch)
    assert outcome.decision.reason == "tier2_miss"
    assert outcome.fallback_reason is None
    assert baseline.calls == 1
    assert runner.calls == []


def test_policy_baseline_and_typed_failure_use_distinct_callbacks():
    deterministic_baseline = _FakeBaseline()

    class FailureFallback:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> str:
            self.calls += 1
            return "failure-fallback"

    failure_fallback = FailureFallback()
    miss = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LLM),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=_FakeRetriever(RetrievalEvidence(tier=LookupTier.MISS)),
        runner=_FakeRunner(),
        baseline=deterministic_baseline,
        failure_fallback=failure_fallback,
        clock=_FakeClock(),
    )
    failed = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LLM),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=_FakeRetriever(_semantic()),
        runner=_FakeRunner(raises=exc.ProviderTimeout("provider timed out")),
        baseline=deterministic_baseline,
        failure_fallback=failure_fallback,
        clock=_FakeClock(),
    )

    assert miss.result == "baseline"
    assert failed.result == "failure-fallback"
    assert deterministic_baseline.calls == 1
    assert failure_fallback.calls == 1


def test_catchable_provider_failure_falls_back_to_baseline():
    retriever = _FakeRetriever(_semantic())
    baseline = _FakeBaseline()
    runner = _FakeRunner(raises=exc.ProviderTimeout("x"))

    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=retriever,
        runner=runner,
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, RetrievalSeededSearch)
    assert outcome.fallback_reason == FallbackReason.PROVIDER_FAILURE
    assert outcome.result == "baseline"
    assert len(runner.calls) == 1
    assert baseline.calls == 1
    assert Phase.FALLBACK in {timing.phase for timing in outcome.timings}


def test_catchable_retrieval_failure_falls_back_to_baseline():
    class FailingRetriever:
        def __call__(self, descriptor: WorkloadDescriptor) -> RetrievalEvidence:
            raise exc.RetrieverUnavailable("index unavailable")

    baseline = _FakeBaseline()
    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=FailingRetriever(),
        runner=_FakeRunner(),
        baseline=baseline,
        clock=_FakeClock(),
    )

    assert isinstance(outcome.decision, BaselineSearch)
    assert outcome.fallback_reason == FallbackReason.RETRIEVAL_FAILURE
    assert outcome.result == "baseline"
    assert baseline.calls == 1
    assert Phase.FALLBACK in {timing.phase for timing in outcome.timings}


def test_non_catchable_provider_failure_propagates():
    runner = _FakeRunner(raises=exc.ProviderAuthError("401"))

    with pytest.raises(exc.ProviderAuthError):
        execute_rag_attempt(
            _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
            probe=_FakeProbe(ExactMiss()),
            describe=lambda: _WORKLOAD,
            retrieve=_FakeRetriever(_semantic()),
            runner=runner,
            baseline=_FakeBaseline(),
            clock=_FakeClock(),
        )


def test_records_phase_timings_and_accumulated_seconds():
    observed: list[PhaseTimingEvent] = []
    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=_FakeRetriever(_semantic()),
        runner=_FakeRunner(),
        baseline=_FakeBaseline(),
        clock=_FakeClock(step=2.0),
        phase_observer=observed.append,
    )

    phases = [event.phase for event in outcome.timings]
    assert phases == [Phase.LOOKUP, Phase.EMBEDDING, Phase.DECISION, Phase.TOTAL]
    assert all(event.seconds > 0 for event in outcome.timings)
    assert outcome.accumulated_seconds > 0
    total_event = next(e for e in outcome.timings if e.phase is Phase.TOTAL)
    assert total_event.seconds == outcome.accumulated_seconds
    assert observed == list(outcome.timings)


def test_deadline_marks_outcome_censored():
    outcome = execute_rag_attempt(
        _treatment(qwen_enabled=True, tuner_mode=TunerMode.LFBO),
        probe=_FakeProbe(ExactMiss()),
        describe=lambda: _WORKLOAD,
        retrieve=_FakeRetriever(_semantic()),
        runner=_FakeRunner(),
        baseline=_FakeBaseline(),
        clock=_FakeClock(),
        deadline_exceeded=lambda: True,
    )

    assert outcome.censored is True
    assert outcome.validated is False
    assert outcome.accumulated_seconds > 0
