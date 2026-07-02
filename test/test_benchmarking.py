from __future__ import annotations

import functools
import itertools
import os
from types import SimpleNamespace

from benchmarks.cute import compare_attention_backends
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
import helion.autotuner.benchmarking as benchmarking
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[1024]))
def _bench_add(x, y):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


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


_FAKE_COMPILER_SEED = {
    "block_sizes": [1, 128, 128],
    "cute_flash_topology": "fa4",
    "cute_flash_causal_lpt_swizzle": 4,
}


def _attention_subprocess_args(**overrides):
    args = SimpleNamespace(
        z=1,
        h=2,
        seq_len=128,
        head_dim=64,
        dtype="float16",
        causal=0,
        biased=1,
        num_runs=5,
        warmup_ms=25,
        rep_ms=100,
        seed=123,
        skip_correctness=0,
        helion_force_flash_config=1,
        helion_force_autotune=0,
        helion_return_lse=0,
        helion_cute_benchmark_timer="wall",
        helion_env=[],
        helion_autotune_effort=None,
        helion_autotune_budget_seconds=None,
        helion_autotune_max_generations=None,
        helion_autotune_best_of_k=None,
        helion_autotune_benchmark_timeout=None,
        helion_autotune_accuracy_check=None,
        helion_autotuner_initial_population=None,
        helion_config=[],
        impls=[],
        stream_subprocesses=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_attention_force_flash_config_uses_compiler_default_seed():
    args = SimpleNamespace(
        helion_config=[],
        helion_force_flash_config=1,
        helion_backend="cute",
    )

    config, overrides = compare_attention_backends._make_helion_config(
        args, _FAKE_COMPILER_SEED
    )

    assert overrides == {}
    assert config == {
        "block_sizes": [1, 128, 128],
        "cute_flash_topology": "fa4",
        "cute_flash_causal_lpt_swizzle": 4,
    }


def test_attention_force_flash_config_applies_manual_overrides_to_seed():
    args = SimpleNamespace(
        helion_config=[("cute_flash_causal_lpt_swizzle", 0)],
        helion_force_flash_config=1,
        helion_backend="cute",
    )

    config, overrides = compare_attention_backends._make_helion_config(
        args, _FAKE_COMPILER_SEED
    )

    assert overrides == {"cute_flash_causal_lpt_swizzle": 0}
    assert config["cute_flash_topology"] == "fa4"
    assert config["cute_flash_causal_lpt_swizzle"] == 0


def test_attention_force_flash_config_falls_back_without_compiler_seed():
    args = SimpleNamespace(
        helion_config=[],
        helion_force_flash_config=1,
        helion_backend="cute",
    )
    config, overrides = compare_attention_backends._make_helion_config(args, None)

    assert overrides == {}
    assert config == {"block_sizes": [1, 128, 128]}


def test_attention_subprocess_forwards_helion_cute_timer():
    args = _attention_subprocess_args(helion_cute_benchmark_timer="event")

    cmd = compare_attention_backends._build_subprocess_cmd(args, "helion-cute")

    flag_index = cmd.index("--helion-cute-benchmark-timer")
    assert cmd[flag_index + 1] == "event"


def test_attention_shape_subprocess_forwards_helion_cute_timer(monkeypatch):
    args = _attention_subprocess_args(helion_cute_benchmark_timer="event")
    seen_cmds = []

    def run_json_subprocess(cmd, args):
        seen_cmds.append(cmd)
        return 0, {"shape": {}, "results": []}, "", ""

    monkeypatch.setattr(
        compare_attention_backends, "_run_json_subprocess", run_json_subprocess
    )

    compare_attention_backends._run_shape_subprocess(
        args, (1, 2, 128, 64, "float16", 0, 1)
    )

    flag_index = seen_cmds[0].index("--helion-cute-benchmark-timer")
    assert seen_cmds[0][flag_index + 1] == "event"


def test_attention_helion_cute_timer_selects_bench_fn():
    calls = []

    def wall_timer(*args, **kwargs):
        return 1.0

    backend = SimpleNamespace(
        get_do_bench=lambda: calls.append("get_do_bench") or wall_timer
    )
    bound = SimpleNamespace(env=SimpleNamespace(backend=backend))

    wall_args = SimpleNamespace(helion_cute_benchmark_timer="wall")
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, wall_args, "cute")
        is wall_timer
    )
    assert calls == ["get_do_bench"]

    event_args = SimpleNamespace(helion_cute_benchmark_timer="event")
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, event_args, "cute")
        is None
    )
    assert (
        compare_attention_backends._helion_do_bench_fn(bound, wall_args, "triton")
        is None
    )
    assert calls == ["get_do_bench"]


def test_attention_markdown_and_wide_csv_include_timer():
    payload = {
        "shape": {
            "z": 1,
            "h": 2,
            "seq_len": 128,
            "head_dim": 64,
            "dtype": "float16",
            "causal": 0,
            "biased": 1,
        },
        "results": [
            {
                "impl": "helion-cute",
                "accuracy": "PASS",
                "benchmark_timer": "event",
                "best_ms": 0.1,
                "median_ms": 0.1,
                "mom_median_ms": 0.1,
                "best_tflops": 1.0,
                "median_tflops": 1.0,
                "mom_median_tflops": 1.0,
            }
        ],
    }

    markdown_rows = compare_attention_backends._markdown_rows(payload)
    wide_rows = compare_attention_backends._wide_rows([payload])

    assert markdown_rows[0]["timer"] == "event"
    assert wide_rows[0]["helion_cute_timer"] == "event"


def test_attention_dense_causal8_suite_uses_larger_shapes():
    shapes = compare_attention_backends._SHAPE_SUITES["dense_causal8"]
    dense_seq_lens = [shape[2] for shape in shapes if shape[5] == 0]
    causal_seq_lens = [shape[2] for shape in shapes if shape[5] == 1]

    assert dense_seq_lens == [32768, 65536, 131072, 262144]
    assert causal_seq_lens == [65536, 131072, 262144, 524288]


def test_attention_autotune_timeout_env_overrides():
    args = SimpleNamespace(
        helion_env=[],
        helion_autotune_effort=None,
        helion_autotune_budget_seconds=None,
        helion_autotune_max_generations=None,
        helion_autotune_best_of_k=None,
        helion_autotune_benchmark_timeout=180,
        helion_autotune_accuracy_check=0,
        helion_autotuner_initial_population=None,
    )

    assert compare_attention_backends._helion_env_overrides(args) == {
        "HELION_AUTOTUNE_BENCHMARK_TIMEOUT": "180",
        "HELION_AUTOTUNE_ACCURACY_CHECK": "0",
    }


def test_attention_gpu_policy_is_opt_in(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", raising=False)

    compare_attention_backends._check_gpu_policy()


def test_attention_gpu_policy_restricts_when_configured(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS", "6,7")

    with pytest.raises(SystemExit):
        compare_attention_backends._check_gpu_policy()


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


_FAKE_EVENT_CLOCK = itertools.count(1)


class _RecordingEvent:
    """torch.cuda.Event stand-in that records which stream it was fired on."""

    def __init__(self, enable_timing=False):
        self.recorded_on = None
        self._t = None

    def record(self, stream=None):
        self.recorded_on = stream
        # Monotonic fake timestamps so elapsed_time is positive and stable.
        self._t = float(next(_FAKE_EVENT_CLOCK))

    def elapsed_time(self, other):
        return abs(other._t - self._t)


class _RecordingStream:
    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


def test_cuda_events_bench_records_on_current_stream_single_sync(monkeypatch):
    """The event bench must record on the current torch stream and sync once
    per timed batch (no per-iteration double sync)."""
    stream = _RecordingStream()
    fake_cuda = SimpleNamespace(
        current_stream=lambda: stream,
        Event=_RecordingEvent,
        is_available=lambda: True,
    )
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    # No-op L2 clearer so we don't touch the real device.
    monkeypatch.setattr(benchmarking, "_make_l2_cache_clearer", lambda: lambda: None)
    # Force a small, deterministic n_repeat regardless of the fake timings.
    monkeypatch.setattr(benchmarking, "sync_object", lambda v, **k: 1.0)

    recorded_streams: list[object] = []
    real_record = _RecordingEvent.record

    def spy_record(self, s=None):
        recorded_streams.append(s)
        return real_record(self, s)

    monkeypatch.setattr(_RecordingEvent, "record", spy_record)

    def fn():
        pass

    benchmarking.do_bench_cuda_events(fn, warmup=1, rep=1, return_mode="median")

    # Every event was recorded on the current torch stream, never the default.
    assert recorded_streams, "no events were recorded"
    assert all(s is stream for s in recorded_streams)
    # Warmup sync + estimate-batch sync + one final timed-batch sync == 3 total,
    # and crucially zero syncs inside the timing loop.
    assert stream.sync_count == 3


@skipIfNotCUDA()
def test_cuda_events_bench_measures_device_time():
    """On a real kernel the event bench should report device time close to
    Triton's do_bench and well below wall-clock (which folds in launch OH)."""
    x = torch.randn(4096, device=DEVICE)
    y = torch.randn(4096, device=DEVICE)
    bound = _bench_add.bind((x, y))
    bound.set_config(helion.Config(block_sizes=[1024]))
    run = bound._run
    for _ in range(5):
        run(x, y)
    torch.cuda.synchronize()
    fn = functools.partial(run, x, y)

    events = benchmarking.do_bench_cuda_events(
        fn, warmup=25, rep=50, return_mode="median"
    )
    triton = benchmarking.do_bench(fn, warmup=25, rep=50, return_mode="median")
    wall = benchmarking.do_bench_generic(fn, warmup=25, rep=50, return_mode="median")

    # Device-time regimes agree closely; both are far below wall-clock, which
    # carries per-call CPU launch overhead.
    assert events > 0
    assert abs(events - triton) <= 0.5 * triton + 5e-3
    assert events < wall
