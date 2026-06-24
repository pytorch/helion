from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

import helion.autotuner.benchmarking as benchmarking


class _FakeStream:
    def wait_stream(self, stream):
        self.waited_stream = stream


class _FakeStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGraphContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGraph:
    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


class _FakeCuda:
    def __init__(self, *, available=True, capturing=False):
        self.available = available
        self.capturing = capturing
        self.current_stream_obj = _FakeStream()
        self.graph_obj = None
        self.synchronize_count = 0

    def is_available(self):
        return self.available

    def is_current_stream_capturing(self):
        return self.capturing

    def Stream(self):
        return _FakeStream()

    def stream(self, stream):
        return _FakeStreamContext()

    def current_stream(self):
        return self.current_stream_obj

    def synchronize(self):
        self.synchronize_count += 1

    def CUDAGraph(self):
        self.graph_obj = _FakeGraph()
        return self.graph_obj

    def graph(self, graph):
        return _FakeGraphContext()


def _fake_torch(cuda):
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=None))


def test_cudagraph_defaults_off(monkeypatch):
    fake_cuda = _FakeCuda()
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)

    def fn():
        return "plain"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn


def test_cudagraph_replay_wraps_callable(monkeypatch):
    fake_cuda = _FakeCuda()
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    calls = []

    def fn():
        calls.append("call")
        return len(calls)

    replay = benchmarking._maybe_cudagraph_replay(fn)

    assert replay() == 2
    assert calls == ["call", "call"]
    assert fake_cuda.graph_obj.replay_count == 1


def test_run_example_enables_cudagraph_only_for_final_benchmark(monkeypatch):
    import helion._testing as testing

    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    seen = []

    def compute_repeat(fn, *, default_cudagraph=False):
        seen.append(("compute_repeat", default_cudagraph))
        return 1

    def interleaved_bench(fns, *, repeat, desc=None, default_cudagraph=False):
        seen.append(("interleaved_bench", default_cudagraph))
        return [1.0, 2.0]

    monkeypatch.setattr(testing, "compute_repeat", compute_repeat)
    monkeypatch.setattr(testing, "interleaved_bench", interleaved_bench)

    testing.run_example(lambda x: x + 1, lambda x: x + 1, (torch.ones(1),))

    assert seen == [("compute_repeat", True), ("interleaved_bench", True)]
    assert "HELION_BENCHMARK_CUDAGRAPH" not in os.environ


def test_cudagraph_auto_falls_back_when_unavailable(monkeypatch):
    fake_cuda = _FakeCuda(available=False)
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")

    def fn():
        return "fallback"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn


def test_cudagraph_auto_skips_nested_capture(monkeypatch):
    fake_cuda = _FakeCuda(capturing=True)
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")

    def fn():
        return "nested"

    assert benchmarking._maybe_cudagraph_replay(fn) is fn


def test_compute_perf_stats_values():
    stats = benchmarking._compute_perf_stats([float(i) for i in range(1, 11)])
    assert stats.min == 1.0
    assert stats.median == 5.5
    assert stats.mean == 5.5
    # numpy "type-7" linear percentile: rank 0.9*(10-1)=8.1 -> 9.0 + 0.1*(10-9)
    assert stats.p90 == pytest.approx(9.1)
    assert stats.std == pytest.approx(3.0276503540974917)
    assert stats.n_samples == 10


def test_compute_perf_stats_empty_is_zeros():
    assert benchmarking._compute_perf_stats([]) == benchmarking.PerfStats(
        0.0, 0.0, 0.0, 0.0, 0.0, 0
    )


def test_compute_perf_stats_single_sample_has_zero_std():
    stats = benchmarking._compute_perf_stats([4.0])
    assert stats.min == stats.median == stats.mean == stats.p90 == 4.0
    assert stats.std == 0.0
    assert stats.n_samples == 1


def test_perf_stats_to_dict_keys():
    stats = benchmarking.PerfStats(1.0, 2.0, 3.0, 4.0, 5.0, 6)
    assert stats.to_dict() == {
        "min": 1.0,
        "median": 2.0,
        "mean": 3.0,
        "p90": 4.0,
        "std": 5.0,
        "n_samples": 6,
    }


def test_summarize_statistics_fallback_stats_mode():
    res = benchmarking._summarize_statistics_fallback([1.0, 2.0, 3.0], None, "stats")
    assert isinstance(res, benchmarking.PerfStats)
    assert res.min == 1.0
    assert res.n_samples == 3
