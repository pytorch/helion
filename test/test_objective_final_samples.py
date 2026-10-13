from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import math
from types import SimpleNamespace

import pytest

from test import test_benchmark_pipeline_objective as fixtures

from helion.autotuner import objective_timing
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.objective_timing import ObjectiveTiming
from helion.autotuner.workload_timer import benchmark_workloads


def test_only_final_uses_fixed_samples(monkeypatch):
    search, factory, calls = fixtures.make(monkeypatch)
    first, second = fixtures.member(1), fixtures.member(2)
    search.benchmark_population([first, second])
    search.population = [first, second]
    search.rebenchmark([first, second], target_ms=27)
    assert all("samples" not in call for call in calls)
    result = search.final_rebenchmark_best(search.best)
    assert result.config == second.config
    assert calls[-1]["samples"] == 120
    assert calls[-1]["warmup_ms"] == 1
    assert calls[-1]["graph_policy"] == "required"
    policy = search._final_rebenchmark_cache_policy()
    assert policy["timing_version"] == 5 and policy["samples"] == 120
    assert ("validate", 2) in factory.events and factory.events[-1][0] == "close"


@pytest.mark.parametrize("samples", [1, 120, 240, None])
def test_explicit_final_policy_and_cache_epoch(monkeypatch, samples):
    original, _, _ = fixtures.make(monkeypatch)
    monkeypatch.setattr(
        fixtures,
        "ObjectiveTiming",
        lambda *args: ObjectiveTiming(*args, final_samples=samples),
    )
    search, _, calls = fixtures.make(monkeypatch)
    members = [fixtures.member(1), fixtures.member(2)]
    search.benchmark_population(members)
    search.population = members
    search.final_rebenchmark_best(search.best)
    assert calls[-1].get("samples") == samples
    assert ("samples" in calls[-1]) == (samples is not None)
    assert (original.objective_benchmark.epoch == search.objective_benchmark.epoch) == (
        samples == 120
    )


@pytest.mark.parametrize("samples", [0, -1, True, 1.5, "120"])
def test_invalid_sample_policy(samples):
    with pytest.raises(BenchmarkObjectiveError, match="final_samples"):
        ObjectiveTiming(lambda: None, "required", "fixture", samples).cache_record()


def test_changed_sample_policy_rejects_stale_epoch(monkeypatch):
    search, _, _ = fixtures.make(monkeypatch)
    object.__setattr__(search.objective_benchmark._timing, "final_samples", 240)
    with pytest.raises(BenchmarkObjectiveError, match="changed"):
        search._final_rebenchmark_cache_policy()


def test_legacy_custom_measure_signature_is_preserved(monkeypatch):
    legacy_calls = []

    class LegacyTiming(ObjectiveTiming):
        def measure(self, workloads, *, warmup_ms, rep_ms):
            legacy_calls.append((warmup_ms, rep_ms))
            return super().measure(workloads, warmup_ms=warmup_ms, rep_ms=rep_ms)

    monkeypatch.setattr(fixtures, "ObjectiveTiming", LegacyTiming)
    search, _, calls = fixtures.make(monkeypatch)
    members = [fixtures.member(1), fixtures.member(2)]
    search.benchmark_population(members)
    search.population = members
    search.final_rebenchmark_best(search.best)
    assert legacy_calls and all("samples" not in call for call in calls)
    assert search._final_rebenchmark_cache_policy()["samples"] is None


def test_nonobjective_policy_is_unchanged(monkeypatch):
    search, _, _ = fixtures.make(monkeypatch, objective=False)
    policy = search._final_rebenchmark_cache_policy()
    assert policy["timing_version"] == 4 and "samples" not in policy


def test_final_late_oracle_failure_is_still_rejected(monkeypatch):
    search, factory, calls = fixtures.make(monkeypatch)
    first, second = fixtures.member(1), fixtures.member(2)
    search.benchmark_population([first, second])
    search.population = [first, second]
    factory.late_bad.add(2)
    assert search.final_rebenchmark_best(search.best).config == first.config
    assert math.isinf(second.perf) and calls[-1]["samples"] == 120


class Runtime:
    def __init__(self):
        self.stream = "caller"
        self.capturing = False
        self.replay = None
        self.events = []

    def current_stream(self):
        return self.stream

    def new_stream(self, caller):
        return "side"

    def set_current_stream(self, stream):
        self.stream = stream

    @contextmanager
    def stream_scope(self, stream):
        old = self.stream
        self.stream = stream
        try:
            yield stream
        finally:
            self.stream = old

    def wait_stream(self, stream, other):
        self.events.append("wait")

    def synchronize(self, stream):
        self.events.append("sync")

    def new_event(self):
        return object()

    def record_event(self, event, stream):
        self.events.append("event")

    def elapsed_ms(self, start, end):
        return 0.0472

    @contextmanager
    def capture(self, stream):
        self.capturing = True
        try:
            yield SimpleNamespace(replay=lambda: self.replay())
        finally:
            self.capturing = False


@pytest.mark.parametrize("graph_policy", ["required", "disabled"])
def test_actual_timer_has_120_samples_and_preserves_boundaries(
    monkeypatch, graph_policy
):
    runtime = Runtime()
    state = [False]

    def reset():
        runtime.events.append("reset")
        state[0] = True

    def flush():
        runtime.events.append("flush")

    def run():
        assert state[0]
        state[0] = False
        runtime.events.append("run")
        if runtime.capturing:
            runtime.replay = run

    workload = BenchmarkWorkload(run, reset, lambda output: None, lambda: None)
    observed = []

    def timer(workloads, **kwargs):
        results = benchmark_workloads(workloads, runtime=runtime, **kwargs)
        observed.extend(results)
        return results

    monkeypatch.setattr(objective_timing, "benchmark_workloads", timer)
    timing = ObjectiveTiming(flush, graph_policy, "CPU-fixture")
    samples = timing.measure(
        [workload], warmup_ms=1, rep_ms=5000, samples=timing.final_rebenchmark_samples
    )
    assert len(samples[0]) == 120
    assert observed[0].warmup_calls == math.ceil(1 / 0.0472) == 22
    assert runtime.events.count("event") == 2 * (120 + 1)
    for index, event in enumerate(runtime.events):
        if event == "event" and runtime.events[index - 1] == "flush":
            assert runtime.events[index - 2 : index + 4] == [
                "reset",
                "flush",
                "event",
                "run",
                "event",
                "sync",
            ]
    assert runtime.current_stream() == "caller"
    assert runtime.events.count("run") == 120 + 22 + 1 + (
        3 if graph_policy == "required" else 1
    )
    assert (
        dataclasses.replace(timing, final_samples=None).final_rebenchmark_samples
        is None
    )
