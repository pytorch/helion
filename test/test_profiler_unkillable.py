"""CPU-only coverage for profiler evidence and unkillable-worker finalization."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable
from helion.autotuner.profiler_timing import CallableIdentity
from helion.autotuner.profiler_timing import ProfilerChunk
from helion.autotuner.profiler_timing import ProfilerTimingObservation
from helion.autotuner.profiler_timing import ProfilerTimingPolicy
from helion.autotuner.search_timing import BenchmarkMeasurement
from helion.autotuner.search_timing import SearchTimingPolicy
from helion.runtime.config import Config
from helion.runtime.settings import Settings


def _measurement(policy, count=3, milliseconds=1):
    chunk = ProfilerChunk(
        count,
        count * milliseconds * 1_000_000,
        count,
        count,
        0,
        1,
        (("cudaLaunchKernel", ("kernel",)),),
    )
    return BenchmarkMeasurement(
        policy,
        ProfilerTimingObservation(
            ProfilerTimingPolicy(), CallableIdentity("compiled", "1" * 64), (chunk,)
        ),
        0,
        count,
    )


@pytest.fixture
def search(monkeypatch):
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    value = object.__new__(PopulationBasedSearch)
    value.timing_policy = SearchTimingPolicy.create()
    value._resolved_timing_policy = value.timing_policy
    value.settings = Settings(
        autotune_timing_method="torch_profiler", autotune_benchmark_subprocess=True
    )
    value.log = Mock()
    value.config_spec = Mock()
    value.config_spec.backend.name = "cute"
    value.config_spec.backend.get_do_bench.return_value = None
    value.config_spec.backend.get_paired_device_micros_bench.return_value = None
    value.kernel = Mock()
    value.kernel.env.device.type = "cuda"
    value.kernel.env.process_group_name = None
    value.kernel.supports_subprocess_benchmark.return_value = True
    value.benchmark_provider = object.__new__(LocalBenchmarkProvider)
    value.population = [
        PopulationMember(
            lambda: None,
            [float(index + 1)],
            (),
            Config(num_warps=2**index),
            "ok",
            measurement=_measurement(value.timing_policy, milliseconds=index + 1),
        )
        for index in range(3)
    ]
    value._unkillable_configs = set()
    value.finishing_rounds = 0
    return value


@pytest.mark.parametrize("mode", ["batch", "mirrored"])
@pytest.mark.parametrize("hung_index", [0, 1, 2])
@pytest.mark.parametrize("finalize", [False, True])
def test_profiler_records_call_local_failure_before_final_fallback(
    search, monkeypatch, mode, hung_index, finalize
):
    members = list(search.population)
    hung = members[hung_index]
    error = BenchmarkWorkerUnkillable("mock worker remained alive")
    # The mirrored provider sees a singleton even for the second/third member.
    error.fn_index = hung_index if mode == "batch" else 0
    seen = []

    def benchmark(fns, **kwargs):
        seen.append((list(fns), kwargs))
        if mode == "batch" or fns == [hung.fn]:
            raise error
        assert search.timing_policy is not None
        return [_measurement(search.timing_policy, kwargs["fixed_repetitions"])]

    monkeypatch.setattr(search.benchmark_provider, "benchmark_isolated", benchmark)

    def run():
        if mode == "batch":
            search._profiler_rebenchmark(
                members, desc="mock", target_ms=6, fresh_process=True
            )
        else:
            search._profiler_mirrored_rebenchmark(members, desc="mock", target_ms=6)

    finishing = Mock(side_effect=AssertionError("must not launch after worker failure"))
    monkeypatch.setattr(search, "run_finishing_phase", finishing)
    monkeypatch.setattr(search, "run_terminal_refinement", finishing)
    monkeypatch.setattr(search, "run_final_pick_verification", finishing)
    if finalize:
        monkeypatch.setattr(search, "final_rebenchmark_best", lambda best: run())
        expected = min((m for m in members if m is not hung), key=lambda m: m.perf)
        assert search._finalize() == expected.config
        assert search.best is expected
    else:
        with pytest.raises(BenchmarkWorkerUnkillable) as caught:
            run()
        assert caught.value is error
    finishing.assert_not_called()
    assert search._unkillable_configs == {hung.config}
    if mode == "batch":
        assert len(seen) == 1
        assert seen[0][0] == [m.fn for m in members]
        assert seen[0][1]["fresh_process"] is True
    else:
        assert [fns for fns, kwargs in seen] == [
            [m.fn] for m in members[: hung_index + 1]
        ]


def test_profiler_finalization_raises_when_no_safe_measured_member(search, monkeypatch):
    members = list(search.population)
    search._unkillable_configs = {m.config for m in members[1:]}
    error = BenchmarkWorkerUnkillable("mock worker remained alive")
    error.fn_index = 0
    benchmark = Mock(side_effect=error)
    monkeypatch.setattr(search.benchmark_provider, "benchmark_isolated", benchmark)
    monkeypatch.setattr(
        search,
        "final_rebenchmark_best",
        lambda best: search._profiler_rebenchmark(members, desc="mock", target_ms=6),
    )
    with pytest.raises(BenchmarkWorkerUnkillable) as caught:
        search._finalize()
    assert caught.value is error
    assert search._unkillable_configs == {m.config for m in members}
    benchmark.assert_called_once()


@pytest.mark.parametrize("fresh_process", [False, True])
def test_profiler_confirmation_preserves_subset_index_and_fresh_process(
    search, monkeypatch, fresh_process
):
    members = list(search.population)
    search.settings.autotune_suspicious_rebenchmark_ratio = 0.5
    error = BenchmarkWorkerUnkillable("mock confirmation worker remained alive")
    error.fn_index = 0
    benchmark = Mock(side_effect=error)
    monkeypatch.setattr(search.benchmark_provider, "benchmark_isolated", benchmark)
    with pytest.raises(BenchmarkWorkerUnkillable) as caught:
        search._confirm_suspicious_rebenchmark_timings(
            members,
            [members[0].perf, members[1].perf / 10, members[2].perf],
            desc="mock confirmation",
            fresh_process=fresh_process,
        )
    assert caught.value is error
    assert search._unkillable_configs == {members[1].config}
    assert benchmark.call_args.args == ([members[1].fn],)
    if fresh_process:
        assert benchmark.call_args.kwargs["fresh_process"] is True
    else:
        assert "fresh_process" not in benchmark.call_args.kwargs
