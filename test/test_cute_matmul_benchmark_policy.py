from __future__ import annotations

from contextlib import contextmanager
import logging
import statistics
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_benchmark_worker import _FakeTimingEvent
from test.test_cute_grouped_fp32_output import _grouped_fp32_output

import helion
from helion._compiler.backend import TritonBackend
from helion._compiler.cute.backend import CuteBackend
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmarking import do_bench
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.runtime.settings import Settings


@pytest.fixture(autouse=True)
def _graph_environment(monkeypatch) -> None:
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)


class _CpuKernel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object) -> None:
        self.calls += 1


class _GraphEvent:
    def __init__(self, runtime, name: str) -> None:
        self.runtime = runtime
        self.name = name
        self.time = 0.0

    def record(self) -> None:
        self.runtime.enqueue("event", self)

    def elapsed_time(self, other) -> float:
        return other.time - self.time


class _GraphStream:
    def __init__(self) -> None:
        self.sync_count = 0

    def wait_stream(self, stream) -> None:
        return

    def synchronize(self) -> None:
        self.sync_count += 1


class _CapturedGraph:
    def __init__(self, runtime, commands) -> None:
        self.runtime = runtime
        self.commands = commands
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        # Submission can leave the GPU idle, before any captured event node.
        self.runtime.time += 7.0
        for kind, value in self.commands:
            self.runtime.execute(kind, value)


class _GraphRuntime:
    """Execute captured device commands without repeating their Python calls."""

    def __init__(self, kernel_ms: float) -> None:
        self.kernel_ms = kernel_ms
        self.time = 0.0
        self.capture = None
        self.graphs = []
        self.events = []
        self.event_kwargs = []
        self.kernel_calls = 0
        self.flushes = 0
        self.stream = _GraphStream()

    def execute(self, kind: str, value) -> None:
        if kind == "event":
            value.time = self.time
        elif kind == "flush":
            self.flushes += 1
            self.time += 3.0
        else:
            assert kind == "kernel"
            self.time += value

    def enqueue(self, kind: str, value) -> None:
        if self.capture is None:
            self.execute(kind, value)
        else:
            self.capture.append((kind, value))

    def kernel(self) -> None:
        self.kernel_calls += 1
        # This host delay occurs during setup/capture, outside graph replay.
        self.time += 40.0
        self.enqueue("kernel", self.kernel_ms)

    def clear_cache(self, cache) -> None:
        self.enqueue("flush", None)

    def event(self, **kwargs):
        self.event_kwargs.append(kwargs)
        event = _GraphEvent(self, "start" if not self.events else "end")
        self.events.append(event)
        return event

    @contextmanager
    def capture_graph(self, **kwargs):
        commands = []
        self.capture = commands
        graph = _CapturedGraph(self, commands)
        self.graphs.append(graph)
        try:
            yield graph
        finally:
            self.capture = None

    @contextmanager
    def stream_context(self, stream):
        yield stream

    @contextmanager
    def patched(self, elapsed_times=()):
        with (
            _event_runtime(list(elapsed_times)) as (clear_cache, synchronize),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.Stream", return_value=self.stream),
            patch("torch.cuda.current_stream", return_value=self.stream),
            patch("torch.cuda.stream", side_effect=self.stream_context),
            patch("torch.cuda.Event", side_effect=self.event),
            patch("helion.runtime.cute_cuda_graph", side_effect=self.capture_graph),
            patch(
                "helion.autotuner.benchmarking.time.perf_counter",
                side_effect=lambda: self.time / 1000.0,
            ),
        ):
            clear_cache.side_effect = self.clear_cache
            yield clear_cache, synchronize


@pytest.fixture
def matmul_bound():
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _forbid_cuda_work(), _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        assert isinstance(bound.config_spec.backend, CuteBackend)
        assert bound.config_spec.matmul_facts
        assert not bound.config_spec.cute_flash_search_enabled
        yield bound


def _provider(bound) -> LocalBenchmarkProvider:
    settings = Settings(
        backend="cute",
        autotune_precompile=None,
        autotune_accuracy_check=False,
        autotune_benchmark_timeout=17,
        autotune_log_level=logging.CRITICAL,
    )
    kernel = SimpleNamespace(
        env=bound.env,
        compile_config=Mock(),
        format_kernel_decorator=lambda config, settings: "kernel()",
    )
    with (
        patch.object(
            LocalBenchmarkProvider, "_compute_baseline", return_value=(None, (), None)
        ),
        patch.object(
            LocalBenchmarkProvider,
            "_compute_effective_tolerances",
            return_value=(0.0, 0.0),
        ),
        patch.object(LocalBenchmarkProvider, "_decide_num_jobs", return_value=1),
    ):
        return LocalBenchmarkProvider(
            kernel,
            settings,
            bound.config_spec,
            (),
            AutotuningLogger(settings),
            AutotuneMetrics(),
        )


@contextmanager
def _event_runtime(elapsed_times: list[float]):
    clear_cache = Mock()
    synchronize = Mock()
    active = SimpleNamespace(
        get_device_interface=lambda: SimpleNamespace(
            Event=lambda **kwargs: _FakeTimingEvent(elapsed_times),
            synchronize=synchronize,
        ),
        get_empty_cache_for_benchmark=lambda: object(),
        clear_cache=clear_cache,
    )
    triton_module = ModuleType("triton")
    triton_module.runtime = SimpleNamespace(driver=SimpleNamespace(active=active))
    testing_module = ModuleType("triton.testing")
    testing_module._summarize_statistics = lambda times, quantiles, return_mode: (
        statistics.median(times)
    )
    with patch.dict(
        sys.modules, {"triton": triton_module, "triton.testing": testing_module}
    ):
        yield clear_cache, synchronize


def test_real_nonflash_matmul_policy_reaches_event_job_and_reuses_long_probe(
    matmul_bound,
) -> None:
    provider = _provider(matmul_bound)
    assert provider._probe_long_kernel()
    assert provider._effective_source_dedup_enabled()
    assert matmul_bound.config_spec.backend.get_do_bench() is None
    provider._precompile_args_path = "unused-args.pt"
    provider._benchmark_worker = Mock()
    provider._benchmark_worker.run.side_effect = lambda job, timeout: job()
    fn = _CpuKernel()
    elapsed_times = [100.0]
    with (
        _event_runtime(elapsed_times) as (clear_cache, synchronize),
        patch(
            "helion.autotuner.benchmark_provider._serialize_compiled_fn",
            return_value=object(),
        ),
        patch("helion.autotuner.benchmark_job._load_compiled_fn", return_value=fn),
        patch(
            "helion.autotuner.benchmark_job.load_trusted_kernel_args", return_value=()
        ),
        patch("helion.autotuner.benchmark_job.do_bench_generic") as wall_timer,
    ):
        result = provider._run_subprocess_benchmark_job(fn, warmup=1, rep=50)
    job = provider._benchmark_worker.run.call_args.args[0]
    assert job.probe_long_kernel is True
    assert job.default_cudagraph is True
    assert job.use_wall_clock is False
    assert provider._benchmark_worker.run.call_args.kwargs["timeout"] == 17
    assert result == 100.0
    assert fn.calls == 2
    clear_cache.assert_called_once()
    assert synchronize.call_count == 2
    wall_timer.assert_not_called()
    assert not elapsed_times


def test_event_long_probe_keeps_five_call_calibration_for_a_short_kernel() -> None:
    fn = _CpuKernel()
    elapsed_times = [1.0, 4.0, 1.0, 1.0, 1.0]
    with (
        _forbid_cuda_work(),
        _event_runtime(elapsed_times) as (clear_cache, synchronize),
    ):
        result = do_bench(
            fn, warmup=1, rep=3, return_mode="median", probe_long_kernel=True
        )
    assert result == 1.0
    assert fn.calls == 10
    assert clear_cache.call_count == 8
    assert synchronize.call_count == 4
    assert not elapsed_times


def test_real_matmul_sources_deduplicate_inert_knobs_and_separate_geometry(
    matmul_bound,
) -> None:
    configs = [
        helion.Config(block_sizes=[128, 16, 16], pid_type="flat"),
        helion.Config(
            block_sizes=[128, 16, 16],
            pid_type="flat",
            tcgen05_ab_stages=4,
            tcgen05_c_stages=4,
            tcgen05_l2_swizzle_size=8,
        ),
        helion.Config(block_sizes=[128, 32, 16], pid_type="flat"),
    ]
    sources = [matmul_bound.to_code(config) for config in configs]
    assert sources[0] == sources[1]
    assert sources[0] != sources[2]
    backend = matmul_bound.config_spec.backend
    functions = [_CpuKernel() for _ in configs]
    for source, fn in zip(sources, functions, strict=True):
        backend.annotate_compiled_module(SimpleNamespace(kernel=fn), source, "kernel")
    hashes = [backend.generated_source_hash(fn) for fn in functions]
    assert hashes[0] == hashes[1] and hashes[0] != hashes[2]
    provider = _provider(matmul_bound)
    provider.kernel.compile_config.side_effect = functions
    with patch.object(
        provider, "_benchmark_function", side_effect=[100.0, 50.0]
    ) as benchmark:
        results = provider.benchmark(configs)
    assert benchmark.call_count == 2
    assert [result.perf for result in results] == [100.0, 100.0, 50.0]
    assert [result.status for result in results] == ["ok", "deduplicated", "ok"]
    assert results[1].fn is results[0].fn
    assert provider._autotune_metrics.num_unique_sources == 2
    assert provider._autotune_metrics.num_source_deduplications == 1
    with patch(
        "helion.autotuner.benchmark_provider.dist.is_initialized", return_value=True
    ):
        assert not provider._effective_source_dedup_enabled()
    pointwise_spec = SimpleNamespace(cute_flash_search_enabled=False, matmul_facts=())
    assert not backend.probe_long_autotune_kernels(pointwise_spec)
    assert not backend.should_deduplicate_generated_sources(pointwise_spec)
    assert not backend.default_autotune_cudagraph(pointwise_spec)
    assert not TritonBackend().default_autotune_cudagraph(matmul_bound.config_spec)


def test_real_matmul_graph_job_uses_internal_events_and_reuses_long_probe(
    matmul_bound,
) -> None:
    provider = _provider(matmul_bound)
    provider.settings.autotune_benchmark_subprocess = True
    provider.kernel.supports_subprocess_benchmark = lambda: True
    with patch(
        "helion.autotuner.benchmark_provider.dist.is_initialized", return_value=False
    ):
        assert provider._subprocess_benchmark_enabled()
    provider._precompile_args_path = "unused-args.pt"
    provider._benchmark_worker = Mock()
    provider._benchmark_worker.run.side_effect = lambda job, timeout: job()
    runtime = _GraphRuntime(kernel_ms=100.0)
    with (
        runtime.patched() as (clear_cache, synchronize),
        patch("helion.autotuner.benchmark_provider._serialize_compiled_fn"),
        patch(
            "helion.autotuner.benchmark_job._load_compiled_fn",
            return_value=runtime.kernel,
        ),
        patch(
            "helion.autotuner.benchmark_job.load_trusted_kernel_args", return_value=()
        ),
        patch("helion.autotuner.benchmark_job.do_bench_generic") as wall_timer,
    ):
        result = provider._run_subprocess_benchmark_job(
            runtime.kernel, warmup=1, rep=50
        )
    job = provider._benchmark_worker.run.call_args.args[0]
    assert job.default_cudagraph and job.probe_long_kernel
    assert job.use_wall_clock is False
    assert provider._benchmark_worker.run.call_args.kwargs["timeout"] == 17
    assert result == 100.0
    assert runtime.kernel_calls == 2  # Setup and capture; replays bypass Python.
    assert runtime.graphs[0].replays == runtime.flushes == 1
    assert [
        (kind, value.name if kind == "event" else None)
        for kind, value in runtime.graphs[0].commands
    ] == [("flush", None), ("event", "start"), ("kernel", None), ("event", "end")]
    assert runtime.event_kwargs == [{"enable_timing": True, "external": True}] * 2
    clear_cache.assert_called_once()
    synchronize.assert_not_called()  # The capture stream is synchronized instead.
    wall_timer.assert_not_called()


@pytest.mark.parametrize("kernel_ms", [0.08, 100.0])
@pytest.mark.parametrize("pre_warmed", [False, True])
def test_graph_policy_keeps_fixed_samples_cold_and_excludes_host_gaps(
    kernel_ms: float, pre_warmed: bool
) -> None:
    runtime = _GraphRuntime(kernel_ms)
    with _forbid_cuda_work(), runtime.patched():
        result = do_bench(
            runtime.kernel,
            return_mode="median",
            default_cudagraph=True,
            fixed_repetitions=3,
            probe_long_kernel=True,
            pre_warmed=pre_warmed,
        )
    assert result == pytest.approx(kernel_ms)
    assert runtime.kernel_calls == 2 - int(pre_warmed)
    assert runtime.graphs[0].replays == runtime.flushes == 3


def test_environment_only_graph_request_retains_legacy_replay_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    with (
        _forbid_cuda_work(),
        _event_runtime([100.0]) as (clear_cache, synchronize),
        patch(
            "helion.autotuner.benchmarking._cudagraph_unavailable_reason",
            return_value=None,
        ),
        patch(
            "helion.autotuner.benchmarking._maybe_cudagraph_replay",
            side_effect=lambda fn, **kwargs: fn,
        ) as legacy_replay,
        patch("helion.autotuner.benchmarking.do_bench_cuda_graph") as event_graph,
    ):
        result = do_bench(
            lambda: None, warmup=1, rep=50, return_mode="median", probe_long_kernel=True
        )
    assert result == 100.0
    legacy_replay.assert_called_once()
    event_graph.assert_not_called()


def test_graph_policy_respects_explicit_environment_disable(monkeypatch) -> None:
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "0")
    runtime = _GraphRuntime(100.0)
    with _forbid_cuda_work(), runtime.patched([100.0]):
        result = do_bench(
            runtime.kernel,
            warmup=1,
            rep=50,
            return_mode="median",
            default_cudagraph=True,
            probe_long_kernel=True,
        )
    assert result == 100.0
    assert runtime.kernel_calls == 2
    assert runtime.graphs == []
    assert runtime.events == []


def test_graph_policy_keeps_gradient_reset_on_the_legacy_path() -> None:
    runtime = _GraphRuntime(100.0)
    parameter = torch.zeros(1, requires_grad=True)
    parameter.grad = torch.ones(1)
    with _forbid_cuda_work(), runtime.patched([100.0]):
        result = do_bench(
            runtime.kernel,
            return_mode="median",
            default_cudagraph=True,
            grad_to_none=[parameter],
            fixed_repetitions=1,
        )
    assert result == 100.0
    assert parameter.grad is None
    assert runtime.kernel_calls == 2
    assert runtime.graphs == []


def test_graph_policy_propagates_capture_failure() -> None:
    runtime = _GraphRuntime(0.08)
    with (
        _forbid_cuda_work(),
        runtime.patched(),
        patch(
            "helion.runtime.cute_cuda_graph", side_effect=RuntimeError("capture failed")
        ),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        do_bench(runtime.kernel, default_cudagraph=True, fixed_repetitions=3)
    assert runtime.kernel_calls == 1
    assert runtime.graphs == []


def test_baseline_reference_uses_the_same_matmul_graph_policy(matmul_bound) -> None:
    child = _provider(matmul_bound)
    child.settings.autotune_baseline_fn = lambda: None
    parent = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
    parent.args = SimpleNamespace(relative_to="baseline")
    with patch(
        "helion.autotuner.benchmark_provider.do_bench", return_value=0.1
    ) as bench:
        result = parent._measure_reference(child, 0)
    assert result == 0.1
    assert bench.call_args.kwargs["default_cudagraph"] is True
