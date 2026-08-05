from __future__ import annotations

import dataclasses
import math
from math import inf
from types import SimpleNamespace

import pytest

from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.candidate_budget import AttemptCategory
from helion.autotuner.rag.instrumentation import InstrumentationCollector
from helion.autotuner.rag.types import Phase
from helion.runtime.config import Config


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _successful_evaluation(
    collector: InstrumentationCollector,
    *,
    config_id: str,
    performance: float,
):
    return collector.record_evaluation(
        config_id=config_id,
        config_repr=f"Config({config_id})",
        candidate_source="retrieval",
        candidate_category="seed",
        compatibility_status="compatible",
        compilation_status="ok",
        compilation_seconds=0.25,
        correctness_status="ok",
        benchmark_status="ok",
        timeout_status="not_timed_out",
        performance=performance,
    )


def test_evaluations_are_ordered_and_track_validated_incumbent():
    collector = InstrumentationCollector(clock=FakeClock(10.0, 11.0, 13.0, 14.0))

    first = _successful_evaluation(collector, config_id="cfg-a", performance=4.0)
    second = _successful_evaluation(collector, config_id="cfg-b", performance=3.0)
    third = _successful_evaluation(collector, config_id="cfg-c", performance=5.0)

    assert collector.evaluations == (first, second, third)
    assert [item.sequence for item in collector.evaluations] == [1, 2, 3]
    assert [item.elapsed_seconds for item in collector.evaluations] == [1.0, 3.0, 4.0]
    assert [item.incumbent_best_perf for item in collector.evaluations] == [
        4.0,
        3.0,
        3.0,
    ]


@pytest.mark.parametrize(
    (
        "compilation_status",
        "correctness_status",
        "benchmark_status",
        "performance",
    ),
    [
        ("error", "not_run", "not_run", 1.0),
        ("ok", "failed", "not_run", 1.0),
        ("ok", "ok", "error", 1.0),
        ("ok", "ok", "ok", math.inf),
        ("ok", "ok", "ok", math.nan),
    ],
)
def test_failed_or_nonfinite_evaluations_do_not_update_incumbent(
    compilation_status,
    correctness_status,
    benchmark_status,
    performance,
):
    collector = InstrumentationCollector(clock=FakeClock(0.0, 1.0, 2.0))
    _successful_evaluation(collector, config_id="good", performance=3.0)

    failed = collector.record_evaluation(
        config_id="failed",
        config_repr="Config(failed)",
        candidate_source="generated",
        candidate_category="search",
        compatibility_status="compatible",
        compilation_status=compilation_status,
        compilation_seconds=math.inf,
        correctness_status=correctness_status,
        benchmark_status=benchmark_status,
        timeout_status="not_timed_out",
        performance=performance,
    )

    assert failed.compilation_status == compilation_status
    assert failed.correctness_status == correctness_status
    assert failed.benchmark_status == benchmark_status
    assert failed.compilation_seconds is None
    assert failed.performance is None
    assert failed.incumbent_best_perf == 3.0


def test_phase_snapshots_are_ordered_and_include_current_incumbent():
    collector = InstrumentationCollector(
        clock=FakeClock(100.0, 101.0, 103.5, 106.0, 110.0)
    )

    lookup = collector.record_phase_transition(Phase.LOOKUP, phase_seconds=1.0)
    _successful_evaluation(collector, config_id="cfg-a", performance=2.5)
    benchmarking = collector.record_phase_transition(
        Phase.BENCHMARKING, phase_seconds=5.0
    )
    total = collector.record_phase_transition(Phase.TOTAL, phase_seconds=4.0)

    assert collector.phase_snapshots == (lookup, benchmarking, total)
    assert lookup.trajectory_sequence == 1
    assert collector.evaluations[0].trajectory_sequence == 2
    assert benchmarking.trajectory_sequence == 3
    assert total.trajectory_sequence == 4
    assert lookup.elapsed_seconds == 1.0
    assert lookup.phase_seconds == 1.0
    assert lookup.incumbent_best_perf is None
    assert benchmarking.elapsed_seconds == 6.0
    assert benchmarking.phase_seconds == 5.0
    assert benchmarking.incumbent_best_perf == 2.5
    assert total.elapsed_seconds == 10.0
    assert total.phase_seconds == 4.0
    assert total.incumbent_best_perf == 2.5


def test_observation_records_are_frozen_and_empty_by_default():
    collector = InstrumentationCollector(clock=FakeClock(0.0, 1.0))

    assert collector.evaluations == ()
    assert collector.phase_snapshots == ()
    snapshot = collector.record_phase_transition(Phase.LOOKUP, phase_seconds=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.phase_seconds = 2.0


def test_nonfinite_or_backwards_clock_is_clamped_deterministically():
    collector = InstrumentationCollector(
        clock=FakeClock(10.0, 12.0, math.inf, 11.0, 14.0)
    )

    first = collector.record_phase_transition(Phase.LOOKUP, phase_seconds=2.0)
    nonfinite = collector.record_phase_transition(
        Phase.EMBEDDING, phase_seconds=math.inf
    )
    backwards = collector.record_phase_transition(Phase.DECISION, phase_seconds=-1.0)
    recovered = collector.record_phase_transition(Phase.GENERATION, phase_seconds=2.0)

    assert [
        (item.elapsed_seconds, item.phase_seconds)
        for item in (first, nonfinite, backwards, recovered)
    ] == [(2.0, 2.0), (2.0, None), (2.0, None), (4.0, 2.0)]


def test_base_search_records_candidate_outcomes_and_phase_transitions():
    configs = [Config(value=1), Config(value=2)]

    class Provider:
        def benchmark(self, candidates, *, desc):
            assert candidates == configs
            assert desc == "Initial population"
            return [
                BenchmarkResult(
                    configs[0],
                    lambda: None,
                    2.0,
                    "ok",
                    0.25,
                    correctness_time=0.125,
                ),
                BenchmarkResult(configs[1], lambda: None, inf, "timeout", 0.5),
            ]

    collector = InstrumentationCollector()
    search = BaseSearch.__new__(BaseSearch)
    search.settings = SimpleNamespace(autotune_config_filter=None)
    search.benchmark_provider = Provider()
    search.log = SimpleNamespace(debug=lambda message: None)
    search.best_perf_so_far = inf
    search._attempt_instrumentation = None
    search._candidate_attempt_categories = dict.fromkeys(
        configs, AttemptCategory.INITIAL_POPULATION
    )
    search._candidate_sources = {
        configs[0]: "retrieval",
        configs[1]: "random_replacement",
    }
    search.set_attempt_instrumentation(collector)

    search.benchmark_batch(configs, desc="Initial population")

    assert [record.candidate_source for record in collector.evaluations] == [
        "retrieval",
        "random_replacement",
    ]
    assert collector.evaluations[0].correctness_status == "ok"
    assert collector.evaluations[0].performance == 2.0
    assert collector.evaluations[1].compilation_status == "timeout"
    assert collector.evaluations[1].timeout_status == "timed_out"
    assert collector.evaluations[1].performance is None
    assert collector.phase_snapshots[1].phase_seconds == 0.125
    assert [snapshot.phase for snapshot in collector.phase_snapshots] == [
        Phase.COMPILATION,
        Phase.CORRECTNESS,
        Phase.BENCHMARKING,
    ]


def test_rebenchmark_measurements_are_recorded_as_evaluations():
    configs = [Config(value=1), Config(value=2)]
    members = [
        PopulationMember(lambda: None, [3.0], [index], config)
        for index, config in enumerate(configs)
    ]
    collector = InstrumentationCollector()
    search = BaseSearch.__new__(BaseSearch)
    search._attempt_instrumentation = collector
    search._candidate_attempt_categories = dict.fromkeys(
        configs, AttemptCategory.GENERATION
    )
    search._candidate_sources = {}

    search._record_rebenchmark_evaluations(members, [1.5, 2.5], elapsed_seconds=0.75)

    assert [record.performance for record in collector.evaluations] == [1.5, 2.5]
    assert collector.incumbent_best_perf == 1.5
    assert collector.phase_snapshots[-1].phase is Phase.BENCHMARKING


def test_isolated_rebenchmark_timeout_is_not_recorded_as_success():
    config = Config(value=1)
    member = PopulationMember(lambda: None, [3.0], [0], config)
    collector = InstrumentationCollector()
    search = BaseSearch.__new__(BaseSearch)
    search._attempt_instrumentation = collector
    search._candidate_attempt_categories = {config: AttemptCategory.INITIAL_POPULATION}
    search._candidate_sources = {}

    search._record_rebenchmark_evaluations(
        [member], [None], elapsed_seconds=1.0, timeout_indices={0}
    )

    record = collector.evaluations[0]
    assert record.benchmark_status == "timeout"
    assert record.timeout_status == "timed_out"
    assert record.performance is None
