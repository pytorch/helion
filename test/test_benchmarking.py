from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace
from typing import Any
from typing import cast

from benchmarks.cute import compare_attention_backends
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
        self.graph_obj: _FakeGraph | None = None
        self.graphs: list[_FakeGraph] = []
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
        graph = _FakeGraph()
        self.graph_obj = graph
        self.graphs.append(graph)
        return graph

    def graph(self, graph):
        return _FakeGraphContext()


def _fake_torch(cuda):
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=None))


def _last_graph(fake_cuda: _FakeCuda) -> _FakeGraph:
    graph = fake_cuda.graph_obj
    assert graph is not None
    return graph


class _FakePerfCounter:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.001
        return self.value


def _patch_generic_bench_for_cudagraph(monkeypatch, *, env_enabled=True):
    fake_cuda = _FakeCuda()
    monkeypatch.setattr(benchmarking, "torch", _fake_torch(fake_cuda))
    if env_enabled:
        monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    else:
        monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    monkeypatch.setattr(benchmarking, "_make_l2_cache_clearer", lambda: lambda: None)
    monkeypatch.setattr(benchmarking, "synchronize_device", lambda result=None: None)
    monkeypatch.setattr(
        benchmarking,
        "sync_object",
        lambda value, process_group_name=None: value,
    )
    monkeypatch.setattr(benchmarking.time, "perf_counter", _FakePerfCounter())
    return fake_cuda


def _fake_device_value(device_type: str = "cuda"):
    return SimpleNamespace(device=SimpleNamespace(type=device_type))


_FAKE_COMPILER_SEED: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_topology": "fa4",
    "cute_flash_causal_lpt_swizzle": 4,
}

_FAKE_FLASH_DEFAULT: dict[str, object] = {
    "block_sizes": [1, 128, 128],
    "cute_flash_topology": "fa4",
    "cute_flash_s_stage": 2,
}


@dataclass
class _FakeConfig:
    config: dict[str, object]


class _FakeConfigSpec:
    def __init__(
        self,
        *,
        compiler_default_config: _FakeConfig | None = None,
        compiler_seed_configs: list[_FakeConfig] | None = None,
        cute_flash_search_enabled: bool = False,
        default_config: dict[str, object] | None = None,
    ) -> None:
        self.compiler_default_config = compiler_default_config
        self.compiler_seed_configs = compiler_seed_configs or []
        self.cute_flash_search_enabled = cute_flash_search_enabled
        self._default_config: dict[str, object] = default_config or {
            "block_sizes": [1, 1, 1]
        }

    def default_config(self) -> _FakeConfig:
        return _FakeConfig(self._default_config)


@dataclass
class _FakeBound:
    config_spec: _FakeConfigSpec


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


def test_attention_force_flash_config_uses_compiler_flash_seed_without_default():
    args = SimpleNamespace(
        helion_config=[],
        helion_force_flash_config=1,
        helion_backend="cute",
    )
    bound = _FakeBound(
        _FakeConfigSpec(
            compiler_seed_configs=[_FakeConfig(_FAKE_COMPILER_SEED)],
            cute_flash_search_enabled=True,
            default_config=_FAKE_FLASH_DEFAULT,
        )
    )

    compiler_config = compare_attention_backends._helion_cute_flash_compiler_config(
        bound,
        "cute",
    )
    config, overrides = compare_attention_backends._make_helion_config(
        args,
        compiler_config,
    )

    assert overrides == {}
    assert compiler_config == _FAKE_COMPILER_SEED
    assert config == _FAKE_COMPILER_SEED


def test_attention_force_flash_config_ignores_non_flash_compiler_seed():
    bound = _FakeBound(
        _FakeConfigSpec(
            compiler_seed_configs=[_FakeConfig({"block_sizes": [16, 16]})],
            cute_flash_search_enabled=False,
        )
    )

    compiler_config = compare_attention_backends._helion_cute_flash_compiler_config(
        bound,
        "cute",
    )

    assert compiler_config is None


def test_attention_force_flash_config_uses_flash_default_without_seed():
    bound = _FakeBound(
        _FakeConfigSpec(
            cute_flash_search_enabled=True,
            default_config=_FAKE_FLASH_DEFAULT,
        )
    )

    compiler_config = compare_attention_backends._helion_cute_flash_compiler_config(
        bound,
        "cute",
    )

    assert compiler_config == _FAKE_FLASH_DEFAULT


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
    assert _last_graph(fake_cuda).replay_count == 1


def test_compute_repeat_generic_uses_cudagraph_when_requested(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    calls = []

    def fn():
        calls.append("call")
        return _fake_device_value()

    repeat = benchmarking.compute_repeat_generic(
        fn,
        target_ms=1.0,
        min_repeat=1,
        max_repeat=10,
        estimate_runs=3,
    )

    assert repeat == 3
    assert calls == ["call", "call", "call"]
    assert _last_graph(fake_cuda).replay_count == 3


def test_compute_repeat_generic_uses_default_cudagraph_when_env_unset(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch, env_enabled=False)
    calls = []

    def fn():
        calls.append("call")
        return _fake_device_value()

    assert "HELION_BENCHMARK_CUDAGRAPH" not in os.environ

    repeat = benchmarking.compute_repeat_generic(
        fn,
        target_ms=1.0,
        min_repeat=1,
        max_repeat=10,
        estimate_runs=3,
        default_cudagraph=True,
    )

    assert repeat == 3
    assert calls == ["call", "call", "call"]
    assert _last_graph(fake_cuda).replay_count == 3


def test_interleaved_bench_generic_uses_cudagraph_when_requested(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    calls = []

    def fn_a():
        calls.append("a")
        return _fake_device_value()

    def fn_b():
        calls.append("b")
        return _fake_device_value()

    times = benchmarking.interleaved_bench_generic([fn_a, fn_b], repeat=2)

    assert times == pytest.approx([1.0, 1.0])
    assert calls == ["a", "b", "a", "a", "b", "b"]
    assert [graph.replay_count for graph in fake_cuda.graphs] == [2, 2]


def test_do_bench_generic_uses_cudagraph_when_requested(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    calls = []

    def fn():
        calls.append("call")
        return _fake_device_value()

    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert calls == ["call", "call", "call"]
    assert _last_graph(fake_cuda).replay_count == 15


def test_do_bench_generic_skips_cudagraph_for_non_cuda_output(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    calls = []

    def fn():
        calls.append("call")
        return _fake_device_value("cpu")

    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert len(calls) == 16
    assert fake_cuda.graph_obj is None


def test_do_bench_generic_cudagraph_for_cuda_side_effect_with_cpu_output(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    cuda_state = _fake_device_value()
    calls = []

    def fn():
        calls.append(cuda_state.device.type)
        return _fake_device_value("cpu")

    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert calls == ["cuda", "cuda", "cuda"]
    assert _last_graph(fake_cuda).replay_count == 15


def test_do_bench_generic_cudagraph_for_cuda_default_with_cpu_output(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    cuda_state = _fake_device_value()
    calls = []

    def fn(state=cuda_state):
        calls.append(state.device.type)
        return _fake_device_value("cpu")

    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert calls == ["cuda", "cuda", "cuda"]
    assert _last_graph(fake_cuda).replay_count == 15


def test_do_bench_generic_cudagraph_for_cuda_callable_state_with_cpu_output(
    monkeypatch,
):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)

    class Fn:
        def __init__(self) -> None:
            self.cuda_state = _fake_device_value()
            self.calls: list[str] = []

        def __call__(self):
            self.calls.append(self.cuda_state.device.type)
            return _fake_device_value("cpu")

    fn = Fn()
    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert fn.calls == ["cuda", "cuda", "cuda"]
    assert _last_graph(fake_cuda).replay_count == 15


def test_do_bench_generic_keeps_backward_path_direct(monkeypatch):
    fake_cuda = _patch_generic_bench_for_cudagraph(monkeypatch)
    calls = []
    grad_target = SimpleNamespace(grad="set")

    def fn():
        calls.append("call")
        return "out"

    timing = benchmarking.do_bench_generic(
        fn,
        warmup=1,
        rep=1,
        grad_to_none=cast("Any", [grad_target]),
        return_mode="median",
    )

    assert timing == pytest.approx(1.0)
    assert len(calls) == 16
    assert fake_cuda.graph_obj is None
    assert grad_target.grad is None


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
