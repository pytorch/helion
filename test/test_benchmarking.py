from __future__ import annotations

import os
from types import SimpleNamespace

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


def test_cute_cudagraph_bench_uses_replay_timer(monkeypatch):
    seen = []

    def replay():
        return "replay"

    def fake_get_replay(fn, *, default_enabled=False):
        seen.append(("get_replay", default_enabled))
        return replay, True

    def fake_do_bench_timed(fn, **kwargs):
        # The timed helper takes no cudagraph flag: it never re-wraps.
        seen.append(("do_bench_timed", fn, "default_cudagraph" in kwargs))
        return 1.25

    def fake_warmup(fns):
        seen.append(("warmup", len(fns)))

    monkeypatch.setattr(benchmarking, "_warmup_fns", fake_warmup)
    monkeypatch.setattr(benchmarking, "_get_cudagraph_replay", fake_get_replay)
    monkeypatch.setattr(benchmarking, "_do_bench_timed", fake_do_bench_timed)

    result = benchmarking.do_bench_cudagraph_generic(
        lambda: "plain",
        return_mode="median",
        default_cudagraph=True,
    )

    assert result == 1.25
    # Eager warmup happens before capture, then the replay is timed directly.
    assert seen == [
        ("warmup", 1),
        ("get_replay", True),
        ("do_bench_timed", replay, False),
    ]


def test_cute_cudagraph_bench_falls_back_to_wall_clock(monkeypatch):
    seen = []

    def fake_get_replay(fn, *, default_enabled=False):
        seen.append(("get_replay", default_enabled))
        return fn, False

    def fake_generic(fn, **kwargs):
        seen.append(("generic", kwargs["default_cudagraph"]))
        return 2.5

    def fake_warmup(fns):
        seen.append(("warmup", len(fns)))

    monkeypatch.setattr(benchmarking, "_warmup_fns", fake_warmup)
    monkeypatch.setattr(benchmarking, "_get_cudagraph_replay", fake_get_replay)
    monkeypatch.setattr(benchmarking, "do_bench_generic", fake_generic)

    result = benchmarking.do_bench_cudagraph_generic(
        lambda: "plain",
        return_mode="median",
        default_cudagraph=True,
    )

    assert result == 2.5
    assert seen == [
        ("warmup", 1),
        ("get_replay", True),
        ("generic", True),
    ]


def test_cute_interleaved_cudagraph_bench_uses_replay_timer(monkeypatch):
    seen = []

    def fake_get_replay(fn, *, default_enabled=False):
        seen.append(("get_replay", default_enabled))
        return fn, True

    def fake_interleaved_timed(fns, *, repeat, desc=None):
        # The timed helper takes no cudagraph flag: it never re-wraps.
        seen.append(("interleaved_timed", len(fns), repeat, desc))
        return [1.0, 2.0]

    def fake_warmup(fns):
        seen.append(("warmup", len(fns)))

    monkeypatch.setattr(benchmarking, "_warmup_fns", fake_warmup)
    monkeypatch.setattr(benchmarking, "_get_cudagraph_replay", fake_get_replay)
    monkeypatch.setattr(
        benchmarking, "_interleaved_bench_timed", fake_interleaved_timed
    )

    result = benchmarking.interleaved_bench_cudagraph_generic(
        [lambda: "a", lambda: "b"],
        repeat=7,
        desc="cg",
        default_cudagraph=True,
    )

    assert result == [1.0, 2.0]
    # All eager launches warm up before any capture.
    assert seen == [
        ("warmup", 2),
        ("get_replay", True),
        ("get_replay", True),
        ("interleaved_timed", 2, 7, "cg"),
    ]
