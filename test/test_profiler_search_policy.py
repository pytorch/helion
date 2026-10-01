"""CPU-only guards and orchestration for the explicit whole-search metric."""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
import statistics
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import Mock

import pytest

from helion.autotuner import benchmarking
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.profiler_timing import CallableIdentity
from helion.autotuner.profiler_timing import ProfilerChunk
from helion.autotuner.profiler_timing import ProfilerTimingCapabilityError
from helion.autotuner.profiler_timing import ProfilerTimingObservation
from helion.autotuner.profiler_timing import ProfilerTimingPolicy
from helion.autotuner.search_timing import BenchmarkMeasurement
from helion.autotuner.search_timing import ProfilerSweepTrace
from helion.autotuner.search_timing import SearchTimingPolicy
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    return SearchTimingPolicy.create()


def _measurement(policy, count=3, identity=None, total_ns=None):
    identity = identity or CallableIdentity("compiled", "1" * 64)
    chunk = ProfilerChunk(
        count,
        total_ns or count * 1000,
        count,
        count,
        0,
        1,
        (("cudaLaunchKernel", ("kernel",)),),
    )
    return BenchmarkMeasurement(
        policy,
        ProfilerTimingObservation(ProfilerTimingPolicy(), identity, (chunk,)),
        0,
        count,
    )


@pytest.mark.parametrize("value", [True, False, None, 1, "event", "", []])
def test_invalid_setting(value):
    with pytest.raises((ValueError, TypeError)):
        Settings(autotune_timing_method=value)


def test_default_setting_bytes():
    default = Settings()
    assert "autotune_timing_method" not in default.to_dict()
    assert "autotune_timing_method" not in repr(default)
    selected = Settings(autotune_timing_method="torch_profiler")
    assert selected.to_dict()["autotune_timing_method"] == "torch_profiler"


def test_policy_pickle_and_runtime_mutation(policy, monkeypatch):
    assert pickle.loads(pickle.dumps(policy)) == policy
    policy.validate()
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    with pytest.raises(ProfilerTimingCapabilityError):
        policy.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("autotune_timing_method", "default"),
        ("autotune_benchmark_subprocess", False),
        ("autotune_accuracy_check", False),
        ("autotune_benchmark_fn", lambda *args: []),
    ],
)
def test_settings_mutation(policy, field, value):
    settings = Settings(
        autotune_timing_method="torch_profiler", autotune_benchmark_subprocess=True
    )
    policy.validate_settings(settings)
    setattr(settings, field, value)
    with pytest.raises(ProfilerTimingCapabilityError):
        policy.validate_settings(settings)


@pytest.mark.parametrize("kind", ["policy", "identity", "count", "zero", "empty"])
def test_measurement_rejects_foreign(policy, kind):
    value = _measurement(policy)
    expected_identity = value.observation.identity
    if kind == "policy":
        value = dataclasses.replace(
            value, policy=dataclasses.replace(policy, version=2)
        )
    elif kind == "identity":
        expected_identity = CallableIdentity("other", "2" * 64)
    elif kind == "count":
        value = _measurement(policy, 2)
    elif kind == "zero":
        value = dataclasses.replace(
            value,
            observation=dataclasses.replace(
                value.observation,
                chunks=(dataclasses.replace(value.observation.chunks[0], total_ns=0),),
            ),
        )
    else:
        value = dataclasses.replace(
            value, observation=dataclasses.replace(value.observation, chunks=())
        )
    with pytest.raises(ProfilerTimingCapabilityError):
        value.validate(policy, expected_identity, 3)


@pytest.mark.parametrize("fixed,probe", [(None, False), (None, True), (7, False)])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("pre_warmed", [False, True])
def test_shared_do_bench_calls(policy, monkeypatch, fixed, probe, graph, pre_warmed):
    import helion.autotuner.search_timing as timing

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1" if graph else "0")
    policy = SearchTimingPolicy.create()

    def run(selected):
        calls = []
        elapsed = [5.0, 4.0] if probe else [5.0]
        elapsed.extend([1.0] * 30)

        class Event:
            def record(self):
                pass

            def elapsed_time(self, other):
                return elapsed.pop(0)

        active = SimpleNamespace(
            get_device_interface=lambda: SimpleNamespace(
                Event=lambda **kwargs: Event(), synchronize=lambda: None
            ),
            get_empty_cache_for_benchmark=lambda: SimpleNamespace(
                numel=lambda: 256 * 1024 * 1024 // 4, element_size=lambda: 4
            ),
            clear_cache=lambda cache: calls.append("clear"),
        )
        triton = ModuleType("triton")
        vars(triton)["runtime"] = SimpleNamespace(driver=SimpleNamespace(active=active))
        testing = ModuleType("triton.testing")
        vars(testing)["_summarize_statistics"] = lambda values, quantiles, mode: (
            statistics.median(values)
        )
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setitem(sys.modules, "triton.testing", testing)

        def capture(fn):
            calls.append("capture")
            fn()

            def replay():
                calls.append("replay")
                return fn()

            return replay

        monkeypatch.setattr(benchmarking, "_cudagraph_unavailable_reason", lambda: None)
        monkeypatch.setattr(benchmarking, "_make_cudagraph_replay", capture)

        def collect(fn, **kwargs):
            for _ in range(kwargs["sample_count"]):
                kwargs["clear"]()
                fn()
            return _measurement(policy, kwargs["sample_count"], kwargs["identity"])

        monkeypatch.setattr(timing, "collect_search_measurement", collect)
        if selected:
            result = benchmarking.do_bench(
                lambda: calls.append("fn"),
                warmup=1,
                rep=3,
                return_mode="mean",
                fixed_repetitions=fixed,
                probe_long_kernel=probe,
                pre_warmed=pre_warmed,
                timing_policy=policy,
                identity=CallableIdentity("x", "1" * 64),
            )
        else:
            result = benchmarking.do_bench(
                lambda: calls.append("fn"),
                warmup=1,
                rep=3,
                return_mode="median",
                fixed_repetitions=fixed,
                probe_long_kernel=probe,
                pre_warmed=pre_warmed,
            )
        return calls, result

    ordinary, ordinary_result = run(False)
    profiled, result = run(True)
    if probe:
        # The default timer returns its long event probe. The profiler must
        # collect one separate, attributed sample after that same estimate.
        sample = ["clear", "replay", "fn"] if graph else ["clear", "fn"]
        assert profiled == ordinary + sample
    else:
        assert profiled == ordinary
    setup_calls = int(not pre_warmed) + int(graph)
    estimated_calls = 0 if fixed is not None else (1 if probe else 5)
    default_timed_calls = fixed or (0 if probe else 4)  # includes one warmup
    assert ordinary.count("fn") == setup_calls + estimated_calls + default_timed_calls
    assert isinstance(ordinary_result, float)
    assert isinstance(result, BenchmarkMeasurement)
    assert result.observation.call_count == (fixed or (1 if probe else 3))


def test_worker_returns_evidence(policy, monkeypatch):
    import helion.autotuner.benchmark_job as jobs

    spec = SerializedCompiledFunction("compiled", "def compiled(): pass", None, None)
    identity = CallableIdentity(
        "compiled", hashlib.sha256(spec.source_code.encode()).hexdigest()
    )
    measurement = _measurement(policy, 7, identity)
    monkeypatch.setattr(jobs, "_load_compiled_fn_for_worker", lambda spec: lambda: None)
    monkeypatch.setattr(jobs, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(jobs, "load_trusted_kernel_args", lambda path: [])
    monkeypatch.setattr(jobs, "_clone_args", lambda args, group: args)
    bench = Mock(return_value=measurement)
    monkeypatch.setattr(jobs, "do_bench", bench)
    result = BenchmarkJob(spec, "unused", fixed_repetitions=7, timing_policy=policy)()
    assert pickle.loads(pickle.dumps(result)) == measurement
    assert bench.call_args.kwargs["return_mode"] == "mean"
    assert bench.call_args.kwargs["identity"] == identity
    assert bench.call_args.kwargs["fixed_repetitions"] == 7


def test_terminal_mean_record(policy):
    from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

    block = _measurement(policy, 3)
    trace = ProfilerSweepTrace([[0]], [[block]], [block.perf], 1.0, block.perf, 1, 3, 3)
    record = LFBOPatternSearch._flash_terminal_trace_metric(["config"], trace)
    assert "median_ms" not in record and "elapsed_ms" not in record
    assert record["mean_kernel_work_ms"] == [
        {"config_id": "config", "value": block.perf}
    ]


def test_profile_rebenchmark_never_calls_default(policy, monkeypatch):
    provider = _provider(policy)
    provider.timing_policy = policy
    provider._resolved_timing_policy = policy
    provider.settings = Settings(
        autotune_timing_method="torch_profiler", autotune_benchmark_subprocess=True
    )
    old = _measurement(policy)
    fresh = _measurement(policy, total_ns=9000)
    fns = [lambda: None, lambda: None]
    provider._timing_measurements = dict.fromkeys(fns, fresh)
    provider.benchmark_isolated = Mock(return_value=[fresh, fresh])
    search = object.__new__(PopulationBasedSearch)
    search.timing_policy = policy
    search._resolved_timing_policy = policy
    search.settings = provider.settings
    search.kernel = provider.kernel
    search.config_spec = provider.config_spec
    search.benchmark_provider = provider
    members = [
        PopulationMember(fn, [old.perf], [], Config(), measurement=old) for fn in fns
    ]
    apply = Mock()
    search._apply_rebenchmark_timings = apply
    search.rebenchmark(members, use_isolated=False, use_interleaved=True)
    provider.benchmark_isolated.assert_called_once()
    assert apply.call_args.kwargs["measurements"] == [fresh, fresh]
    assert apply.call_args.args[1] == [fresh.perf, fresh.perf]


def _provider(policy):
    from helion.autotuner.metrics import AutotuneMetrics

    value = object.__new__(LocalBenchmarkProvider)
    value.timing_policy = policy
    value._resolved_timing_policy = policy
    value.settings = Settings(
        autotune_timing_method="torch_profiler",
        autotune_benchmark_subprocess=True,
        autotune_ignore_errors=True,
    )
    value.kernel = Mock(
        supports_subprocess_benchmark=Mock(return_value=True),
        env=SimpleNamespace(
            process_group_name=None, device=SimpleNamespace(type="cuda")
        ),
    )
    value.config_spec = Mock(
        backend=SimpleNamespace(
            name="cute",
            get_do_bench=lambda: None,
            get_paired_device_micros_bench=lambda: None,
            probe_long_autotune_kernels=lambda _config_spec: False,
        ),
        compiler_seed_timeout_retry_repetitions=None,
        cute_flash_search_enabled=False,
    )
    value.mutated_arg_indices = []
    value._args_unpicklable = False
    value._subprocess_wrapper_unloadable = False
    value._precompile_args_path = "unused"
    value._benchmark_worker = Mock()
    value._timing_measurements = {}
    value._autotune_metrics = AutotuneMetrics()
    value.log = Mock()
    return value


@pytest.mark.parametrize("phase", ["initial", "subprocess", "rebenchmark", "accuracy"])
def test_capability_error_never_prunes(policy, phase):
    value = _provider(policy)
    failure = ProfilerTimingCapabilityError("missing launch correlation")
    value._run_subprocess_benchmark_job = Mock(side_effect=failure)
    if phase == "accuracy":
        value._run_subprocess_benchmark_job = Mock(return_value=0.001)
        value._run_subprocess_accuracy_check_job = Mock(side_effect=failure)
    with pytest.raises(ProfilerTimingCapabilityError, match="missing launch"):
        if phase == "initial":
            value._benchmark_function(Config(), lambda: None)
        elif phase in ("subprocess", "accuracy"):
            value._benchmark_function_subprocess(Config(), lambda: None)
        else:
            value.benchmark_isolated([lambda: None], warmup=1, rep=1)
    assert isinstance(value.log, Mock)
    value.log.record_timing_failure.assert_called_once_with(failure)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_args_unpicklable", True),
        ("_subprocess_wrapper_unloadable", True),
        ("mutated_arg_indices", [0]),
    ],
)
def test_lost_provider_capability(policy, attribute, value):
    provider = _provider(policy)
    setattr(provider, attribute, value)
    with pytest.raises(ProfilerTimingCapabilityError):
        provider._subprocess_benchmark_enabled()


@pytest.mark.parametrize(
    "mode", ["missing_args", "serialize", "load", "float", "identity"]
)
def test_worker_transport_fails_closed(policy, monkeypatch, mode):
    from helion.autotuner.benchmark_job import CompiledFunctionLoadError
    import helion.autotuner.benchmark_provider as module

    provider = _provider(policy)
    spec = SerializedCompiledFunction("compiled", "def compiled(): pass", None, None)
    serialize = Mock(return_value=spec)
    monkeypatch.setattr(module, "_serialize_compiled_fn", serialize)
    assert isinstance(provider._benchmark_worker, Mock)
    if mode == "missing_args":
        provider._precompile_args_path = None
    elif mode == "serialize":
        serialize.side_effect = RuntimeError("unavailable")
    elif mode == "load":
        provider._benchmark_worker.run.side_effect = CompiledFunctionLoadError(
            "unavailable"
        )
    elif mode == "float":
        provider._benchmark_worker.run.return_value = 0.001
    else:
        provider._benchmark_worker.run.return_value = _measurement(policy)
    with pytest.raises(ProfilerTimingCapabilityError):
        provider._run_subprocess_benchmark_job(lambda: None, warmup=1, rep=1)
    assert provider._timing_measurements == {}


def test_terminal_weighted_sweeps(policy, monkeypatch):
    provider = _provider(policy)
    fns: list[Callable[..., object]] = [lambda: None, lambda: None]
    measurements = dict.fromkeys(fns, _measurement(policy))
    monkeypatch.setattr(provider, "measurement_for", lambda fn: measurements[fn])
    visited = []

    def benchmark(fns, **kwargs):
        fn = fns[0]
        visited.append(fn)
        count = kwargs["fixed_repetitions"]
        measurements[fn] = _measurement(
            policy, count, total_ns=count * (1000 + len(visited))
        )
        return [measurements[fn]]

    monkeypatch.setattr(provider, "benchmark_isolated", benchmark)
    search = object.__new__(PopulationBasedSearch)
    search.timing_policy = policy
    search._resolved_timing_policy = policy
    search.settings = provider.settings
    search.kernel = provider.kernel
    search.config_spec = provider.config_spec
    search.benchmark_provider = provider
    search._apply_rebenchmark_timings = Mock()
    search.log = Mock()
    monkeypatch.setattr(search, "_repeat_for_target_ms", lambda *args: 130)
    old = _measurement(policy)
    members = [
        PopulationMember(fn, [old.perf], [], Config(), measurement=old) for fn in fns
    ]
    trace = search.mirrored_rebenchmark(members, desc="terminal", target_ms=1)
    assert isinstance(trace, ProfilerSweepTrace)
    assert trace.total_calls == 132
    assert trace.orders[0] == [0, 1] and trace.orders[1] == [1, 0]
    for index, blocks in enumerate(trace.observations):
        expected = sum(block.observation.total_ns for block in blocks) / 132 / 1_000_000
        assert trace.means_ms[index] == expected
        selected = search._apply_rebenchmark_timings.call_args.kwargs["measurements"][
            index
        ]
        assert selected is not None
        assert selected.perf == expected


@pytest.mark.parametrize(
    "change",
    [
        "backend",
        "device",
        "distributed",
        "group",
        "callback",
        "accuracy",
        "subprocess",
        "custom_provider",
        "aot",
        "backend_timer",
        "paired_timer",
        "unloadable",
        "active",
    ],
)
def test_upfront_capability_before_scoring(policy, monkeypatch, change):
    from helion.autotuner.base_search import BaseSearch
    from helion.autotuner.finite_search import FiniteSearch

    provider = _provider(policy)
    kernel = provider.kernel
    assert isinstance(kernel, Mock)
    kernel.settings = provider.settings
    kernel.config_spec = provider.config_spec
    kernel.env.device = SimpleNamespace(type="cuda")
    provider_cls = LocalBenchmarkProvider
    if change == "backend":
        monkeypatch.setattr(kernel.config_spec.backend, "name", "triton")
    elif change == "device":
        kernel.env.device.type = "cpu"
    elif change == "distributed":
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    elif change == "group":
        kernel.env.process_group_name = "ranks"
    elif change == "callback":
        kernel.settings.autotune_benchmark_fn = Mock()
    elif change == "accuracy":
        kernel.settings.autotune_baseline_accuracy_check_fn = Mock()
    elif change == "subprocess":
        kernel.settings.autotune_benchmark_subprocess = False
    elif change == "custom_provider":
        provider_cls = Mock()
    elif change == "aot":
        kernel.settings.autotune_cache = "AOTAutotuneCache"
    elif change == "backend_timer":
        kernel.config_spec.backend.get_do_bench = lambda: Mock()
    elif change == "paired_timer":
        kernel.config_spec.backend.get_paired_device_micros_bench = lambda: Mock()
    elif change == "unloadable":
        kernel.supports_subprocess_benchmark.return_value = False
    else:
        monkeypatch.setattr("torch.autograd._profiler_enabled", lambda: True)
    search = object.__new__(FiniteSearch)
    with pytest.raises(ProfilerTimingCapabilityError):
        BaseSearch.__init__(search, kernel, (), provider_cls)
        provider._check_timing_capability()


def test_multishape_rejected_before_provider_construction(policy):
    from helion.autotuner.base_search import BaseSearch
    from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
    from helion.autotuner.finite_search import FiniteSearch

    provider = _provider(policy)
    kernel = provider.kernel
    assert isinstance(kernel, Mock)
    kernel.settings = provider.settings
    kernel.config_spec = provider.config_spec
    args = object.__new__(_MultiShapeAutotuneArgs)
    with pytest.raises(ProfilerTimingCapabilityError):
        BaseSearch.__init__(
            object.__new__(FiniteSearch), kernel, cast("Sequence[object]", args)
        )


@pytest.mark.parametrize("stage", ["unavailable", "args", "serialize", "load"])
def test_accuracy_cannot_fall_back(policy, monkeypatch, stage):
    from helion.autotuner.benchmark_job import CompiledFunctionLoadError
    import helion.autotuner.benchmark_provider as providers

    provider = _provider(policy)
    provider._precompile_baseline_path = "baseline"
    provider._effective_atol = provider._effective_rtol = 1e-3
    provider._scale_atol = False
    assert isinstance(provider._benchmark_worker, Mock)
    monkeypatch.setattr(
        providers,
        "_serialize_compiled_fn",
        Mock(return_value=SerializedCompiledFunction("x", "def x(): pass", None, None)),
    )
    if stage == "unavailable":
        monkeypatch.setattr(
            provider, "_subprocess_accuracy_check_enabled", lambda: False
        )
    elif stage == "args":
        provider._precompile_baseline_path = None
    elif stage == "serialize":
        monkeypatch.setattr(
            providers,
            "_serialize_compiled_fn",
            Mock(side_effect=RuntimeError("unloadable")),
        )
    else:
        provider._benchmark_worker.run.side_effect = CompiledFunctionLoadError(
            "missing"
        )
    with pytest.raises(ProfilerTimingCapabilityError):
        provider._run_subprocess_accuracy_check_job(lambda: None)


@pytest.mark.parametrize("field", ["stream_id", "device_index", "work_signature"])
def test_aggregated_chunk_identity_is_preserved(policy, field):
    value = _measurement(policy, 2)
    first = value.observation.chunks[0]
    second = dataclasses.replace(first)
    object.__setattr__(
        second, field, (("other", ("other",)),) if field == "work_signature" else 9
    )
    changed = dataclasses.replace(
        value,
        fixed_repetitions=4,
        observation=dataclasses.replace(value.observation, chunks=(first, second)),
    )
    with pytest.raises(ProfilerTimingCapabilityError):
        changed.validate(policy)


def test_singleton_terminal_reuses_earlier_evidence(policy):
    search = _search(policy)
    search.timing_policy = policy
    search._resolved_timing_policy = policy
    search.settings = _provider(policy).settings
    value = _measurement(policy)
    member = PopulationMember(
        lambda: None, [value.perf], [], Config(), measurement=value
    )
    trace = search.mirrored_rebenchmark([member], desc="singleton")
    assert isinstance(trace, ProfilerSweepTrace)
    assert trace.orders == [] and trace.total_calls == 0
    assert trace.means_ms == [value.perf]
    member.perfs[-1] *= 2
    with pytest.raises(ProfilerTimingCapabilityError):
        search.mirrored_rebenchmark([member], desc="singleton")


@pytest.mark.parametrize("failed", [False, True])
def test_external_training_domain_and_source(policy, monkeypatch, failed):
    from helion.autotuner.benchmark_provider import BenchmarkResult
    import helion.autotuner.precompile_future as precompile
    from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

    serialized = SerializedCompiledFunction("x", "def x(): pass", None, None)
    monkeypatch.setattr(precompile, "_serialize_compiled_fn", lambda fn: serialized)
    identity = CallableIdentity(
        "x", hashlib.sha256(serialized.source_code.encode()).hexdigest()
    )
    value = _measurement(policy, identity=identity)
    search = object.__new__(LFBOPatternSearch)
    search.timing_policy = policy
    search._resolved_timing_policy = policy
    search.settings = _provider(policy).settings
    provider = _provider(policy)
    search.kernel = provider.kernel
    search.config_spec = provider.config_spec
    search.config_gen = Mock()
    search._append_training_sample = Mock()
    result = BenchmarkResult(
        Config(),
        lambda: None,
        float("inf") if failed else value.perf,
        "error" if failed else "ok",
        None,
        None if failed else value,
        policy,
    )
    search.seed_training_data([result])
    search._append_training_sample.assert_called_once()
    for foreign in (None, dataclasses.replace(policy, version=2)):
        with pytest.raises(ProfilerTimingCapabilityError):
            search.seed_training_data([result._replace(timing_policy=foreign)])
    if not failed:
        monkeypatch.setattr(
            precompile,
            "_serialize_compiled_fn",
            lambda fn: dataclasses.replace(serialized, source_code="different"),
        )
        with pytest.raises(ProfilerTimingCapabilityError):
            search.seed_training_data([result])


def _search(policy):
    provider = _provider(policy)
    search = object.__new__(PopulationBasedSearch)
    search.timing_policy = policy
    search._resolved_timing_policy = policy
    search.settings = provider.settings
    search.log = provider.log
    search.config_spec = provider.config_spec
    search.kernel = provider.kernel
    search.args = ()
    search.benchmark_provider = provider
    search._benchmarked_members = {}
    search._pinned_finalist_configs = set()
    search._pinned_finalist_members = {}
    search._terminal_refinement_members = {}
    search.best_perf_so_far = float("inf")
    search._search_space_tracker = None
    search._autotune_budget_start = None
    return search


def test_initial_history_finalist_pin_rebenchmark(policy, monkeypatch):
    from helion.autotuner.benchmark_provider import BenchmarkResult

    search = _search(policy)
    old = _measurement(policy)

    def fn():
        return None

    config = Config(num_warps=4)
    monkeypatch.setattr(
        search,
        "_apply_config_filter",
        lambda configs: (configs, list(range(len(configs)))),
    )
    monkeypatch.setattr(
        search.benchmark_provider,
        "benchmark",
        Mock(return_value=[BenchmarkResult(config, fn, old.perf, "ok", None, old)]),
    )
    result = search.benchmark_batch([config], desc="initial")[0]
    assert result.timing_policy == policy and result.measurement is old
    member = PopulationMember(fn, [old.perf], [], config, "ok", measurement=old)
    search._pinned_finalist_configs.add(config)
    monkeypatch.setattr(search, "_final_rebenchmark_top_k", lambda: 4)
    search._record_benchmarked_member(member)
    assert search._pinned_finalist_members[config].measurement is old
    assert search._terminal_refinement_members is not None
    assert search._terminal_refinement_members[config].measurement is old
    # Snapshot histories must not acquire later mutable entries by aliasing.
    member.measurements.append(_measurement(policy, total_ns=6000))
    assert search._pinned_finalist_members[config].measurements == [old]
    fresh = _measurement(policy, total_ns=9000)
    monkeypatch.setattr(
        search.benchmark_provider,
        "benchmark_isolated",
        Mock(return_value=[fresh, fresh]),
    )
    monkeypatch.setattr(search.benchmark_provider, "measurement_for", lambda fn: fresh)
    second = PopulationMember(
        fn, [old.perf], [], Config(num_warps=8), "ok", measurement=old
    )
    search.settings.autotune_suspicious_rebenchmark_ratio = 0
    search.rebenchmark(
        [member, second], use_isolated=False, use_interleaved=True, desc="finalist"
    )
    assert member.measurement is fresh and member.perf == fresh.perf
    assert search._pinned_finalist_members[config].measurement is fresh
    assert search._terminal_refinement_members is not None
    assert search._terminal_refinement_members[config].measurement is fresh


@pytest.mark.parametrize(
    "phase",
    ["initial", "ordinary", "suspicious", "terminal", "outer_trial", "outer_rebench"],
)
def test_whole_route_errors_escape(policy, monkeypatch, phase):
    from helion.autotuner.base_cache import AutotuneCacheBase
    from helion.autotuner.local_cache import LocalAutotuneCache

    failure = ProfilerTimingCapabilityError("sentinel attribution failure")
    search = _search(policy)
    old = _measurement(policy)
    members = [
        PopulationMember(
            lambda: None, [old.perf], [], Config(num_warps=w), "ok", measurement=old
        )
        for w in (4, 8)
    ]
    monkeypatch.setattr(
        search.benchmark_provider, "benchmark_isolated", Mock(side_effect=failure)
    )
    with pytest.raises(ProfilerTimingCapabilityError, match="sentinel"):
        if phase == "initial":
            monkeypatch.setattr(
                search, "_apply_config_filter", lambda configs: (configs, [0])
            )
            monkeypatch.setattr(
                search.benchmark_provider, "benchmark", Mock(side_effect=failure)
            )
            search.benchmark_batch([Config()])
        elif phase == "ordinary":
            search.rebenchmark(members, use_isolated=False)
        elif phase == "suspicious":
            search.settings.autotune_suspicious_rebenchmark_ratio = 0.5
            search._confirm_suspicious_rebenchmark_timings(
                members, [old.perf / 10] * 2, desc="confirm"
            )
        elif phase == "terminal":
            search.mirrored_rebenchmark(members, desc="terminal")
        else:
            cache = object.__new__(LocalAutotuneCache)
            cache.autotuner = search
            trial = Mock(timing_policy=policy, log=Mock())
            trial.autotune.side_effect = failure
            trial.benchmark_batch.side_effect = failure
            cache._autotuner_factory = lambda: trial
            if phase == "outer_trial":
                AutotuneCacheBase._run_one_trial(cache, 0, 2, 0, skip_cache=True)
            else:
                cache._run_rebench_round([Config()])


@pytest.mark.parametrize("which", ["trial", "rebench"])
def test_outer_factory_rejects_changed_domain(policy, which):
    from helion.autotuner.base_cache import AutotuneCacheBase
    from helion.autotuner.local_cache import LocalAutotuneCache

    cache = object.__new__(LocalAutotuneCache)
    cache.autotuner = _search(policy)
    trial = Mock(timing_policy=None)
    cache._autotuner_factory = lambda: trial
    with pytest.raises(ProfilerTimingCapabilityError, match="domain"):
        if which == "trial":
            AutotuneCacheBase._run_one_trial(cache, 0, 2, 0, skip_cache=True)
        else:
            cache._run_rebench_round([Config()])
    trial.autotune.assert_not_called()
    trial._prepare.assert_not_called()


@pytest.mark.parametrize("kind", ["attribution", "collector_api", "callable"])
def test_failed_collector_preserves_trace_and_callable_semantics(
    policy, monkeypatch, kind
):
    import helion.autotuner.profiler_timing as collector
    from helion.autotuner.search_timing import collect_search_measurement

    failure = (
        ValueError("candidate body")
        if kind == "callable"
        else (
            ProfilerTimingCapabilityError("correlation")
            if kind == "attribution"
            else RuntimeError("unsupported profiler")
        )
    )
    identity = CallableIdentity("fn", "source")
    trace = collector.ProfilerTrace(identity, 1, ())

    def collect(fn, **kwargs):
        kwargs["trace_sink"](trace)
        if kind == "callable":
            fn()
        raise failure

    monkeypatch.setattr(collector, "collect_profiler_timing", collect)

    def fn():
        raise failure

    expected = ValueError if kind == "callable" else ProfilerTimingCapabilityError
    with pytest.raises(expected) as caught:
        collect_search_measurement(
            fn,
            policy=policy,
            identity=identity,
            sample_count=1,
            clear=lambda: None,
            synchronize=lambda: None,
            warmup_calls=0,
            fixed_repetitions=1,
        )
    if kind == "callable":
        assert caught.value is failure and caught.value.args == ("candidate body",)
    else:
        import json

        assert json.loads(caught.value.args[-1])["trace"]["call_count"] == 1


def test_final_pick_uses_isolated_profiler_means(policy, monkeypatch):
    search = _search(policy)
    old = _measurement(policy)
    fresh = _measurement(policy, total_ns=6000)
    members = [
        PopulationMember(
            lambda: None, [old.perf], [], Config(num_warps=w), "ok", measurement=old
        )
        for w in (4, 8)
    ]
    search.population = members
    search._benchmarked_members = {m.config: m for m in members}
    search._pinned_finalist_configs.add(members[0].config)
    search._pinned_finalist_members[members[0].config] = members[0]
    monkeypatch.setattr(
        search.config_spec.backend,
        "generated_source_hash",
        lambda fn: None,
        raising=False,
    )
    monkeypatch.setattr(
        search.benchmark_provider,
        "benchmark_isolated",
        Mock(return_value=[fresh, fresh]),
    )
    monkeypatch.setattr(search.benchmark_provider, "measurement_for", lambda fn: fresh)
    search.settings.autotune_suspicious_rebenchmark_ratio = 0
    selected = search.final_rebenchmark_best(members[0])
    assert selected.measurement is fresh
    assert selected.perf == fresh.perf


@pytest.mark.parametrize("drop_selector", [False, True])
@pytest.mark.parametrize(
    "route", ["initial", "history", "rebench", "terminal", "provider", "accuracy"]
)
def test_resolved_policy_cannot_be_dropped(policy, drop_selector, route):
    search = _search(policy)
    search.timing_policy = None
    provider = search.benchmark_provider
    assert isinstance(provider, LocalBenchmarkProvider)
    provider.timing_policy = None
    if drop_selector:
        search.settings.autotune_timing_method = "default"
    value = _measurement(policy)
    member = PopulationMember(
        lambda: None, [value.perf], [], Config(), "ok", measurement=value
    )
    with pytest.raises(ProfilerTimingCapabilityError, match="removed"):
        if route == "initial":
            search.benchmark_batch([])
        elif route == "history":
            search._record_benchmarked_member(member)
        elif route == "rebench":
            search.rebenchmark([])
        elif route == "terminal":
            search.mirrored_rebenchmark([], desc="terminal")
        elif route == "provider":
            provider._run_subprocess_benchmark_job(lambda: None, warmup=1, rep=1)
        else:
            provider._run_subprocess_accuracy_check_job(lambda: None)


@pytest.mark.parametrize(
    "cache_name",
    [
        "LocalAutotuneCache",
        "StrictLocalAutotuneCache",
        "RemoteAutotuneCache",
        "StrictRemoteAutotuneCache",
    ],
)
@pytest.mark.parametrize("flash", [False, True])
def test_cache_domain_is_structural_and_default_unchanged(
    policy, monkeypatch, cache_name, flash
):
    import torch

    from helion.autotuner import cache_classes
    import helion.autotuner.local_cache as local

    cls = cache_classes[cache_name]
    assert issubclass(cls, local.LocalAutotuneCache)

    def key(selected):
        cache = object.__new__(cls)
        cache.args = ()
        cache.kernel = Mock()
        cache.kernel.kernel.fn = test_default_setting_bytes
        cache.kernel.kernel._create_bound_kernel_cache_key.return_value = (
            SimpleNamespace(
                specialization_key=(32, 128), extra_results=(), compiler_seed_results=()
            )
        )
        cache.kernel.config_spec.cache_fingerprint_hash.return_value = "spec"
        cache.kernel.config_spec.cute_flash_search_enabled = flash
        cache.kernel.env.backend.name = "cute"
        cache.kernel.extra_cache_key.return_value = ""
        cache.autotuner = Mock(
            timing_policy=selected,
            settings=Settings(
                autotune_timing_method="torch_profiler" if selected else "default"
            ),
        )
        cache.autotuner.cache_policy.return_value = {"version": 1}
        return cache._generate_key()

    monkeypatch.setattr(local, "extract_device", lambda args: torch.device("cuda:0"))
    monkeypatch.setattr(local, "get_device_name", lambda dev: "CPU-only metadata")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ordinary = key(None)
    selected = key(policy)
    assert ordinary.stable_hash() != selected.stable_hash()
    assert (
        dataclasses.replace(selected, search_policy_hash=ordinary.search_policy_hash)
        == ordinary
    )
    assert key(None).stable_hash() == ordinary.stable_hash()
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    assert key(SearchTimingPolicy.create()).stable_hash() != selected.stable_hash()


@pytest.mark.parametrize("operation", ["autotune", "get", "put"])
def test_cache_revalidates_before_io(policy, monkeypatch, operation):
    from helion.autotuner.local_cache import LocalAutotuneCache

    cache = object.__new__(LocalAutotuneCache)
    cache.autotuner = _search(policy)
    get = Mock()
    if operation == "autotune":
        monkeypatch.setattr(cache, "get", get)
    cache.autotuner.timing_policy = None
    cache.autotuner.settings.autotune_timing_method = "default"
    with pytest.raises(ProfilerTimingCapabilityError):
        if operation == "autotune":
            cache.autotune()
        elif operation == "get":
            cache.get()
        else:
            cache.put(Config())
    get.assert_not_called()


@pytest.mark.parametrize(
    "change", ["backend", "device", "timer", "paired", "subprocess"]
)
def test_finalization_rejects_mutated_backend_before_selection(
    policy, monkeypatch, change
):
    search = _search(policy)
    if change == "backend":
        monkeypatch.setattr(search.config_spec.backend, "name", "triton")
    elif change == "device":
        monkeypatch.setattr(search.kernel.env, "device", SimpleNamespace(type="cpu"))
    elif change == "timer":
        monkeypatch.setattr(search.config_spec.backend, "get_do_bench", lambda: Mock())
    elif change == "paired":
        monkeypatch.setattr(
            search.config_spec.backend, "get_paired_device_micros_bench", lambda: Mock()
        )
    else:
        monkeypatch.setattr(
            search.kernel, "supports_subprocess_benchmark", lambda: False
        )
    select = Mock()
    monkeypatch.setattr(search, "final_rebenchmark_best", select)
    with pytest.raises(ProfilerTimingCapabilityError, match="capability"):
        search._finalize()
    select.assert_not_called()
