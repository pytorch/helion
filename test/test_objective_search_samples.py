from __future__ import annotations

import math

import pytest

from test import test_benchmark_pipeline_objective as fixtures
from test.test_objective_final_samples import Runtime

from helion.autotuner import objective_timing
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.objective_timing import ObjectiveTiming
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.workload_timer import benchmark_workloads


def make(monkeypatch, *, search_samples=30, final_samples=120, graph="required"):
    monkeypatch.setattr(
        fixtures,
        "ObjectiveTiming",
        lambda flush, policy, tag: ObjectiveTiming(
            flush, graph, tag, final_samples, search_samples
        ),
    )
    return fixtures.make(monkeypatch)


@pytest.mark.parametrize("final_samples", [120, None])
@pytest.mark.parametrize("search_samples", [30, None])
def test_real_initial_main_polish_and_final_are_separate(
    monkeypatch, search_samples, final_samples
):
    search, _, calls = make(
        monkeypatch, search_samples=search_samples, final_samples=final_samples
    )
    first, second, third = (fixtures.member(i) for i in (1, 2, 3))
    search.benchmark_initial_population([first])
    search.benchmark_population([second])
    # The dispatch must not infer final policy from the human-readable label.
    search.rebenchmark([first, second], desc="Final verification custom", target_ms=27)
    search.population = [first]
    second = fixtures.member(2)
    search.polish_rounds = 2
    monkeypatch.setattr(
        PatternSearch,
        "_generate_neighbors",
        lambda self, flat: {(1,): [(2,)], (2,): [(3,)]}[flat],
    )
    monkeypatch.setattr(
        search, "make_unbenchmarked", lambda flat: {(2,): second, (3,): third}[flat]
    )
    monkeypatch.setattr(search, "should_rebenchmark", lambda member: True)
    search._polish_descent({first.config})
    assert all(c.get("samples") == search_samples for c in calls)
    assert all(("samples" in c) == (search_samples is not None) for c in calls)
    assert any(
        p == "Polish round 1: verifying"
        for p, _ in search.objective_benchmark.observations
    )
    chosen = search.final_rebenchmark_best(search.best)
    assert chosen.config == second.config
    assert calls[-1].get("samples") == final_samples
    assert ("samples" in calls[-1]) == (final_samples is not None)
    assert all(c["warmup_ms"] == 1 for c in calls)
    assert search._final_rebenchmark_cache_policy()["samples"] == final_samples


def test_legacy_custom_measure_keeps_signature_and_ms(monkeypatch):
    class LegacyTiming(ObjectiveTiming):
        def measure(self, workloads, *, warmup_ms, rep_ms):
            return super().measure(workloads, warmup_ms=warmup_ms, rep_ms=rep_ms)

    monkeypatch.setattr(
        fixtures, "ObjectiveTiming", lambda *a: LegacyTiming(*a, search_samples=30)
    )
    search, _, calls = fixtures.make(monkeypatch)
    members = [fixtures.member(1), fixtures.member(2)]
    search.benchmark_population(members)
    search.population = members
    search.rebenchmark(members)
    search.final_rebenchmark_best(search.best)
    assert calls and all("samples" not in c for c in calls)
    assert search.objective_benchmark._timing.cache_record()["search_samples"] is None


@pytest.mark.parametrize("samples", [0, -1, True, 1.5, "30"])
def test_invalid_budget(samples):
    with pytest.raises(BenchmarkObjectiveError, match="search_samples"):
        ObjectiveTiming(
            lambda: None, "required", "test", search_samples=samples
        ).cache_record()


@pytest.mark.parametrize("samples", [None, 30, 60])
def test_epoch_binds_budget_and_version(monkeypatch, samples):
    original, _, _ = make(monkeypatch)
    current, _, _ = make(monkeypatch, search_samples=samples)
    assert (
        current.objective_benchmark.epoch == original.objective_benchmark.epoch
    ) == (samples == 30)
    record = current.objective_benchmark._timing.cache_record()
    assert record["search_policy_version"] == 1
    assert record["search_samples"] == samples
    object.__setattr__(current.objective_benchmark._timing, "search_samples", 31)
    with pytest.raises(BenchmarkObjectiveError, match="changed"):
        current.objective_benchmark.validate()


@pytest.mark.parametrize("late", [False, True])
def test_before_and_after_oracles_still_reject(monkeypatch, late):
    search, factory, calls = make(monkeypatch)
    (factory.late_bad if late else factory.bad).add(2)
    members = [fixtures.member(1), fixtures.member(2)]
    search.benchmark_population(members)
    assert math.isinf(members[1].perf)
    assert members[1].status == "accuracy_error"
    assert calls and all(c["samples"] == 30 for c in calls)
    assert factory.events[-1][0] == "close"


def test_actual_timer_search_count_and_full_boundary(monkeypatch):
    search, factory, _ = make(monkeypatch, graph="disabled")
    runtime = Runtime()
    observed = []

    def timer(workloads, **kwargs):
        result = benchmark_workloads(workloads, runtime=runtime, **kwargs)
        observed.extend(result)
        return result

    monkeypatch.setattr(objective_timing, "benchmark_workloads", timer)
    member = fixtures.member(1)
    search.benchmark_population([member])
    assert len(member.objective_measurement.samples_ms) == 30
    assert len(observed[0].samples_ms) == 30
    assert observed[0].warmup_calls == math.ceil(1 / 0.0472)
    assert factory.events.count(("validate", 1)) == 2
    assert factory.events[-1] == ("close", 1)
    for index, event in enumerate(factory.events):
        if event == ("run", 1):
            previous = factory.events[index - 1]
            if previous == ("flush",):
                previous = factory.events[index - 2]
            assert previous == ("reset", 1)
