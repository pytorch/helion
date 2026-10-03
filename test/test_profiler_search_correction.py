from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock

import pytest

from test.test_profiler_search_policy import _measurement
from test.test_profiler_search_policy import _search

from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_worker import BenchmarkSubprocessError
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.profiler_timing import ProfilerTimingCapabilityError
from helion.autotuner.search_timing import SearchTimingPolicy
from helion.runtime.config import Config

if TYPE_CHECKING:
    from helion.autotuner.benchmark_provider import LocalBenchmarkProvider


@pytest.mark.parametrize("confirm", [False, True])
def test_profiler_finalists_cannot_inherit_worker_state(monkeypatch, confirm):
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    search.settings.autotune_suspicious_rebenchmark_ratio = 0.5 if confirm else 0
    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    prior_worker = Mock(samples=100)
    provider._benchmark_worker = prior_worker
    workers: list[Mock] = []
    initial = _measurement(policy, total_ns=30000)
    members = [
        PopulationMember(
            lambda: None,
            [initial.perf],
            [],
            Config(num_warps=warps),
            measurement=initial,
        )
        for warps in (4, 8)
    ]

    def run_job(fn, **kwargs):
        if provider._benchmark_worker is None:
            worker = Mock(samples=0)
            provider._benchmark_worker = worker
            workers.append(worker)
        worker = cast("Mock", provider._benchmark_worker)
        worker.samples += 1
        value = _measurement(policy, total_ns=3000 * worker.samples)
        provider._timing_measurements[fn] = value
        return value.perf

    monkeypatch.setattr(provider, "_run_subprocess_benchmark_job", run_job)
    search.rebenchmark(
        members,
        use_isolated=False,
        confirm_suspicious=confirm,
        candidate_private_args=True,
    )

    fresh = _measurement(policy, total_ns=3000)
    assert [member.perf for member in members] == [fresh.perf, fresh.perf]
    assert [member.measurement for member in members] == [fresh, fresh]
    assert len(workers) == (4 if confirm else 2)
    assert all(worker.samples == 1 for worker in workers)
    prior_worker.shutdown.assert_called_once_with()
    for worker in workers:
        worker.shutdown.assert_called_once_with()
    assert provider._benchmark_worker is None


def test_fresh_profiler_worker_closes_when_fixed_score_has_invalid_evidence(
    monkeypatch,
):
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    prior_worker = cast("Mock", provider._benchmark_worker)
    worker = Mock()

    def run_job(fn, *, warmup, rep, fixed_repetitions):
        assert fixed_repetitions == 7
        provider._benchmark_worker = worker
        value = _measurement(policy, fixed_repetitions)
        provider._timing_measurements[fn] = value
        return value.perf / 2

    monkeypatch.setattr(provider, "_run_subprocess_benchmark_job", run_job)
    with pytest.raises(ProfilerTimingCapabilityError, match="disagrees"):
        provider.benchmark_isolated(
            [lambda: None],
            warmup=0,
            rep=1,
            fresh_process=True,
            fixed_repetitions=7,
        )
    prior_worker.shutdown.assert_called_once_with()
    worker.shutdown.assert_called_once_with()
    assert provider._benchmark_worker is None


@pytest.mark.parametrize("confirm", [False, True])
def test_alias_callable_keeps_each_invocation_and_suspicious_subset(
    monkeypatch, confirm
):
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    search.settings.autotune_suspicious_rebenchmark_ratio = 0.5 if confirm else 0

    def fn():
        return None

    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    initial = _measurement(policy, total_ns=30000)
    members = [
        PopulationMember(
            fn,
            [initial.perf],
            [],
            Config(num_warps=w),
            "deduplicated",
            measurement=initial,
        )
        for w in (4, 8)
    ]
    first = _measurement(policy, total_ns=6000)
    second = _measurement(policy, total_ns=24000)
    confirmation = _measurement(policy, total_ns=9000)
    sequence = [first, second, confirmation] if confirm else [first, second]
    calls = []

    def worker(actual, **kwargs):
        assert actual is fn
        value = sequence[len(calls)]
        calls.append(kwargs)
        provider._timing_measurements[fn] = value
        return value.perf

    monkeypatch.setattr(
        search.benchmark_provider, "_run_subprocess_benchmark_job", worker
    )
    search.rebenchmark(
        members, desc="alias", confirm_suspicious=confirm, use_isolated=False
    )
    expected = [confirmation if confirm else first, second]
    assert [member.measurement for member in members] == expected
    assert [member.perf for member in members] == [value.perf for value in expected]
    assert len(calls) == (3 if confirm else 2)
    if confirm:
        assert calls[2]["warmup"] == 25 and calls[2]["rep"] == 100


@pytest.mark.parametrize("phase", ["initial", "accuracy", "rebenchmark"])
def test_policy_drop_is_fatal_before_failure_accounting(monkeypatch, phase):
    search = _search(SearchTimingPolicy.create())
    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    provider._compile_failure_config_ids = []

    def drop(*args, **kwargs):
        provider.timing_policy = None
        provider.settings.autotune_timing_method = "default"
        provider._validate_timing_policy()

    if phase == "accuracy":
        monkeypatch.setattr(
            provider, "_run_subprocess_benchmark_job", lambda *a, **kw: 0.001
        )
        monkeypatch.setattr(provider, "_run_subprocess_accuracy_check_job", drop)
    else:
        monkeypatch.setattr(provider, "_run_subprocess_benchmark_job", drop)
    with pytest.raises(ProfilerTimingCapabilityError):
        if phase == "rebenchmark":
            provider.benchmark_isolated([lambda: None], warmup=1, rep=1)
        else:
            provider._benchmark_function_subprocess(Config(), lambda: None)
    assert provider._compile_failure_config_ids == []


@pytest.mark.parametrize("phase", ["initial", "accuracy", "rebenchmark"])
@pytest.mark.parametrize(
    "error", [BenchmarkTimeout, BenchmarkSubprocessError, ValueError]
)
def test_policy_drop_precedes_worker_error_accounting(monkeypatch, phase, error):
    search = _search(SearchTimingPolicy.create())
    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    failures = []

    def drop(*args, **kwargs):
        provider.timing_policy = None
        provider.settings.autotune_timing_method = "default"
        raise error("worker failed after policy removal")

    monkeypatch.setattr(
        provider, "_record_worker_failure", lambda *a: failures.append(a)
    )
    monkeypatch.setattr(
        provider, "_record_compile_failure", lambda *a: failures.append(a)
    )
    monkeypatch.setattr(
        provider, "_claim_compiler_seed_timeout_retry", lambda *a: failures.append(a)
    )
    if phase == "accuracy":
        monkeypatch.setattr(
            provider, "_run_subprocess_benchmark_job", lambda *a, **kw: 0.001
        )
        monkeypatch.setattr(provider, "_run_subprocess_accuracy_check_job", drop)
    else:
        monkeypatch.setattr(provider, "_run_subprocess_benchmark_job", drop)
    with pytest.raises(ProfilerTimingCapabilityError):
        if phase == "rebenchmark":
            provider.benchmark_isolated([lambda: None], warmup=1, rep=1)
        else:
            provider._benchmark_function_subprocess(Config(), lambda: None)
    assert failures == []
    assert provider._autotune_metrics.num_isolated_rebenchmark_timeouts == 0


def test_whole_batch_score_check_precedes_all_state_changes():
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    value = _measurement(policy)
    members = [
        PopulationMember(
            lambda: None,
            [value.perf],
            [],
            Config(num_warps=w),
            "ok",
            measurement=value,
            measurements=[value],
        )
        for w in (4, 8)
    ]
    old_best = search.best_perf_so_far
    with pytest.raises(ProfilerTimingCapabilityError):
        search._apply_rebenchmark_timings(members, [value.perf, value.perf / 2])
    assert [member.perfs for member in members] == [[value.perf], [value.perf]]
    assert all(member.measurements == [value] for member in members)
    assert search.best_perf_so_far == old_best
    assert search._terminal_refinement_members == {}


@pytest.mark.parametrize("history", [False, True])
def test_rejected_history_is_not_published_or_extended(history):
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    value = _measurement(policy)
    perfs = [value.perf / 2, value.perf] if history else [value.perf / 2]
    member = PopulationMember(
        lambda: None, perfs, [], Config(), "ok", measurement=value
    )
    with pytest.raises(ProfilerTimingCapabilityError):
        search._record_benchmarked_member(member)
    assert search._terminal_refinement_members == {}
    assert member.measurements == []


def test_isolated_score_cannot_disagree_with_same_call_evidence(monkeypatch):
    policy = SearchTimingPolicy.create()
    search = _search(policy)
    value = _measurement(policy)

    def fn():
        return None

    provider = cast("LocalBenchmarkProvider", search.benchmark_provider)
    provider._timing_measurements[fn] = value
    monkeypatch.setattr(
        search.benchmark_provider,
        "_run_subprocess_benchmark_job",
        lambda *a, **kw: value.perf / 2,
    )
    with pytest.raises(ProfilerTimingCapabilityError):
        search.benchmark_provider.benchmark_isolated([fn], warmup=1, rep=1)
