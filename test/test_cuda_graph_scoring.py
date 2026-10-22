from __future__ import annotations

from contextlib import contextmanager
import gc
import math
import pickle
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from helion.autotuner import cuda_graph_scoring
from helion.autotuner import graph_scoring as graph
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import _WorkerFinalSelectionMeasurer
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmarking import MirroredBenchmarkTrace
from helion.autotuner.metrics import AutotuneMetrics
import helion.runtime as helion_runtime
from helion.runtime.config import Config
from helion.runtime.settings import Settings


def _indexed_check(arguments, output, baseline):
    values, indices = output
    x, lengths, k = arguments
    assert values.shape == indices.shape == (x.shape[0], k)
    for row, length in enumerate(lengths.tolist()):
        selected = indices[row][indices[row] >= 0].long()
        assert selected.numel() == min(length, k)
        assert selected.unique().numel() == selected.numel()
        assert (selected < length).all()
        assert (indices[row][indices[row] < 0] == -1).all()
        assert torch.isneginf(values[row][indices[row] < 0]).all()
        expected = x[row, selected].contiguous().view(torch.uint8)
        actual = values[row][indices[row] >= 0].contiguous().view(torch.uint8)
        assert torch.equal(actual, expected), "payload bits"
        ranks = x[row, :length].double().sort(descending=True).values[:k]
        torch.testing.assert_close(
            values[row][indices[row] >= 0].double().sort(descending=True).values, ranks
        )


def _baseline(x):
    return x + 1


def _ignore_inputs(pristine, arguments):
    pass


def _mutating_baseline(x):
    x.add_(1)
    return x


def _fake_timer(monkeypatch, *, replay=True, captured=True):
    capture = {"active": False}
    calls = []
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capture["active"]
    )
    forbidden = Mock(side_effect=AssertionError("eager timing"))
    monkeypatch.setattr(graph.benchmarking, "do_bench", forbidden)
    monkeypatch.setattr(graph.benchmarking, "do_bench_generic", forbidden)

    def timer(fn, **options):
        calls.append(options)
        fn()
        options["phase"]("capture")
        capture["active"] = captured
        output = fn()
        capture["active"] = False
        original = pickle.loads(pickle.dumps(output))
        options["phase"]("warmup")
        if replay:
            for left, right in zip(
                torch.utils._pytree.tree_leaves(output),
                torch.utils._pytree.tree_leaves(original),
                strict=True,
            ):
                left.copy_(right)
        options["phase"]("measure")
        return [0.003, 0.002, 0.004]

    monkeypatch.setattr(graph.benchmarking, "do_bench_cuda_graph", timer)
    return calls, forbidden


def _job(tmp_path, args, baseline, **options):
    args_path, baseline_path = tmp_path / "args.pt", tmp_path / "baseline.pt"
    torch.save(args, args_path)
    torch.save(baseline, baseline_path)
    return graph.GraphBenchmarkJob(
        benchmark=BenchmarkJob(fn_spec="first", args_path=str(args_path)),
        baseline_path=str(baseline_path),
        **options,
    )


def _evidence(**options):
    return {
        "timer": graph.TIMER,
        "graph_scoring_policy_version": 1,
        "eager_fallback": False,
        "cold_l2_cache": True,
        "graph_output_accuracy": "ok",
        "preserved_inputs": "ok",
        "captures": 1,
        "sample_count": 3,
        "poisoned_outputs": 1,
        "latency_ms": 0.003,
        **options,
    }


@pytest.fixture
def provider(monkeypatch):
    result = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
    result.settings = Settings(autotune_handoff=False)
    result.log = Mock()
    result.kernel = SimpleNamespace(
        env=SimpleNamespace(process_group_name=None),
        supports_subprocess_benchmark=lambda: True,
    )
    result.config_spec = SimpleNamespace(
        backend=SimpleNamespace(name="cute", get_do_bench=lambda: None)
    )
    result.args = (torch.ones(2),)
    result.mutated_arg_indices = ()
    result._graph_scoring = graph.CudaGraphScoring(graph.GraphScoringPolicy())
    result._precompile_args_path = "arguments.pt"
    result._precompile_baseline_path = "baseline.pt"
    result._benchmark_worker = Mock()
    result._benchmark_worker.run.return_value = _evidence()
    result._effective_atol = result._effective_rtol = 0.0
    result._scale_atol = False
    result._autotune_metrics = AutotuneMetrics()
    result._compile_failure_config_ids = []
    result._worker_failure_config_ids = []
    monkeypatch.setattr(
        result, "_compiler_seed_timeout_retry_repetitions", lambda: None
    )
    monkeypatch.setattr(graph, "_has_cuda_tensors", lambda arguments: True)
    monkeypatch.setattr(graph, "_serialize_compiled_fn", lambda fn: "native-wrapper")
    return result


def test_context_callbacks_are_picklable_and_nested_contexts_restore():
    assert graph.current_cuda_graph_scoring() is None
    with cuda_graph_scoring(
        check_output=_indexed_check, poison_output=graph.poison_indexed_output
    ) as outer:
        assert graph.current_cuda_graph_scoring() is outer
        recovered = pickle.loads(pickle.dumps(outer.policy))
        assert recovered.check_output is _indexed_check
        assert recovered.poison_output is graph.poison_indexed_output
        assert recovered.check_inputs is graph.assert_preserved_input_bits
        with cuda_graph_scoring() as inner:
            assert graph.current_cuda_graph_scoring() is inner
        assert graph.current_cuda_graph_scoring() is outer
    assert graph.current_cuda_graph_scoring() is None


def test_local_callback_is_rejected_before_worker_submission():
    with (
        pytest.raises(graph.GraphScoringUnavailable, match="module-level|imported"),
        cuda_graph_scoring(check_output=lambda *args: None),
    ):
        pytest.fail("unimportable callback was accepted")


def test_strict_timer_retains_and_poisons_one_capture(monkeypatch):
    calls, forbidden = _fake_timer(monkeypatch)
    result, output = graph.strict_benchmark(
        lambda: torch.full((2,), 3.0), warmup=17, rep=43, fixed_repetitions=7
    )
    assert result["latency_ms"] == 0.003
    assert result["sample_count"] == 3
    assert result["captures"] == result["poisoned_outputs"] == 1
    assert result["phases"] == ["capture", "warmup", "measure"]
    assert torch.equal(output, torch.full((2,), 3.0))
    assert {key: value for key, value in calls[0].items() if key != "phase"} == {
        "return_mode": "all",
        "warmup": 17,
        "rep": 43,
        "fixed_repetitions": 7,
    }
    forbidden.assert_not_called()


@pytest.mark.parametrize("captured", [False, True])
def test_missing_capture_or_noop_replay_never_supplies_a_score(
    monkeypatch, tmp_path, captured
):
    _fake_timer(monkeypatch, replay=False, captured=captured)
    monkeypatch.setattr(graph, "_has_cuda_tensors", lambda arguments: True)
    monkeypatch.setattr(
        graph, "_load_compiled_fn_for_worker", lambda spec: lambda x: x + 1
    )
    unloaded = Mock()
    monkeypatch.setattr(graph, "_unload_compiled_fn", unloaded)
    job = _job(tmp_path, (torch.arange(2.0),), torch.arange(2.0) + 1)
    with pytest.raises(AssertionError if captured else graph.GraphScoringUnavailable):
        job()
    unloaded.assert_called_once()


@pytest.mark.parametrize("empty", [False, True])
def test_indexed_replay_checks_both_tensors_and_empty_padding(
    monkeypatch, tmp_path, empty
):
    arguments = (
        torch.tensor([[0.0, -0.0, 3.0, 2.0]]),
        torch.tensor([0 if empty else 4], dtype=torch.int32),
        2,
    )
    indices = torch.tensor([[-1, -1] if empty else [3, 2]], dtype=torch.int32)
    values = (
        torch.full((1, 2), -float("inf"))
        if empty
        else arguments[0].gather(1, indices.long())
    )
    output = (values, indices)
    calls, forbidden = _fake_timer(monkeypatch)
    result, replayed = graph.strict_benchmark(
        lambda: (values.clone(), indices.clone()),
        warmup=1,
        rep=2,
        poison=graph.poison_indexed_output,
    )
    assert result["poisoned_outputs"] == 2
    job = _job(
        tmp_path,
        arguments,
        output,
        policy=graph.GraphScoringPolicy(
            check_output=_indexed_check, poison_output=graph.poison_indexed_output
        ),
    )
    job._check(arguments, arguments, replayed)
    forbidden.assert_not_called()
    _fake_timer(monkeypatch, replay=False)
    _, stale = graph.strict_benchmark(
        lambda: (values.clone(), indices.clone()),
        warmup=1,
        rep=2,
        poison=graph.poison_indexed_output,
    )
    with pytest.raises(AssertionError):
        job._check(arguments, arguments, stale)


def test_input_bits_reject_signed_zero_mutation_and_accept_nan():
    original = (torch.tensor([float("nan"), -0.0]),)
    preserved = (original[0].clone(),)
    graph.assert_preserved_input_bits(original, preserved)
    preserved[0][1] = 0.0
    with pytest.raises(AssertionError, match="preserved input"):
        graph.assert_preserved_input_bits(original, preserved)


def test_graph_baseline_preserves_offset_views_and_exact_read_only_bits(
    provider, monkeypatch
):
    storage = torch.arange(24.0)
    arguments = (storage[8:].view(4, 4),)
    arguments[0][0, 0] = float("nan")
    arguments[0][0, 1] = -0.0
    settings = Settings(autotune_baseline_fn=_baseline, autotune_handoff=False)
    monkeypatch.setattr(LocalBenchmarkProvider, "_decide_num_jobs", lambda self: 1)
    with cuda_graph_scoring():
        result = LocalBenchmarkProvider(
            provider.kernel,
            settings,
            provider.config_spec,
            arguments,
            provider.log,
            AutotuneMetrics(),
        )
    assert result.mutated_arg_indices == ()
    assert result._baseline_post_args[0].storage_offset() == 8
    graph.assert_preserved_input_bits(arguments, result._baseline_post_args)


def test_custom_input_callback_cannot_waive_read_only_contract(
    provider, monkeypatch, tmp_path
):
    settings = Settings(autotune_baseline_fn=_mutating_baseline, autotune_handoff=False)
    monkeypatch.setattr(LocalBenchmarkProvider, "_decide_num_jobs", lambda self: 1)
    with (
        pytest.raises(AssertionError, match="preserved input"),
        cuda_graph_scoring(check_inputs=_ignore_inputs),
    ):
        LocalBenchmarkProvider(
            provider.kernel,
            settings,
            provider.config_spec,
            (torch.zeros(2),),
            provider.log,
            AutotuneMetrics(),
        )
    job = _job(
        tmp_path,
        (torch.zeros(2),),
        torch.ones(2),
        policy=graph.GraphScoringPolicy(check_inputs=_ignore_inputs),
    )
    with pytest.raises(AssertionError, match="preserved input"):
        job._check((torch.ones(2),), (torch.zeros(2),), torch.ones(2))


@pytest.mark.parametrize(
    "samples,captures",
    [([], 1), ([0], 1), ([-1], 1), ([float("nan")], 1), ([float("inf")], 1), ([1], 0)],
)
def test_invalid_graph_samples_are_rejected(samples, captures):
    with pytest.raises(ValueError):
        graph.timing_result(samples, captures=captures)


@pytest.mark.parametrize(
    "key,value",
    [
        ("eager_fallback", True),
        ("graph_output_accuracy", None),
        ("preserved_inputs", None),
        ("captures", 0),
        ("poisoned_outputs", 0),
    ],
)
def test_unverified_worker_results_are_rejected(key, value):
    with pytest.raises(graph.GraphScoringUnavailable):
        graph.checked_result(_evidence(**{key: value}))


@pytest.mark.parametrize("repeat", [5, 6, 130])
def test_mirrored_samples_keep_balanced_orders_and_device_medians(repeat):
    class Sampler:
        def __init__(self, value):
            self.value, self.calls = value, 0

        def sample(self):
            self.calls += 1
            return self.value

    samplers = [Sampler(0.003), Sampler(0.001), Sampler(0.002)]
    after = []
    trace = graph.mirrored_samples(samplers, repeat=repeat, after_call=after.append)
    assert all(
        trace.orders[index + 1] == trace.orders[index][::-1]
        for index in range(0, len(trace.orders), 2)
    )
    assert [sampler.calls for sampler in samplers] == [trace.total_calls] * 3
    assert len(after) == trace.total_calls * 3
    assert trace.medians_ms == pytest.approx([0.003, 0.001, 0.002])
    for order, values in zip(trace.orders, trace.elapsed_ms, strict=True):
        assert values == pytest.approx([samplers[index].value for index in order])


def test_native_candidate_dispatch_serializes_policy_and_never_runs_parent_accuracy(
    provider, monkeypatch
):
    forbidden = Mock(side_effect=AssertionError("parent candidate execution"))
    monkeypatch.setattr(provider, "_run_subprocess_accuracy_check_job", forbidden)
    assert provider._benchmark_function(Config(), forbidden) == 0.003
    job = provider._benchmark_worker.run.call_args.args[0]
    assert isinstance(job, graph.GraphBenchmarkJob)
    assert (
        pickle.loads(pickle.dumps(job)).policy.check_inputs
        is graph.assert_preserved_input_bits
    )
    assert job.baseline_path == "baseline.pt"
    assert provider._graph_scoring.evidence == [_evidence()]
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("capture failed"),
        AssertionError("replay oracle failed"),
        BenchmarkTimeout("timing failed"),
    ],
)
def test_failed_native_graph_candidates_never_run_in_parent(provider, failure):
    provider._benchmark_worker.run.side_effect = failure
    forbidden = Mock(side_effect=AssertionError("parent candidate execution"))
    assert math.isinf(provider._benchmark_function(Config(), forbidden))
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("capture failed"),
        AssertionError("replay oracle failed"),
        BenchmarkTimeout("timing failed"),
    ],
)
def test_failed_finalist_graph_jobs_disqualify_previous_scores(provider, failure):
    provider._benchmark_worker.run.side_effect = failure
    assert provider.benchmark_isolated([Mock()], warmup=1, rep=3) == [
        IsolatedBenchmarkFailure("error")
    ]


def test_native_refinement_forces_worker_even_when_interleaved_requested(
    provider, monkeypatch
):
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.benchmark_provider, search.settings, search.kernel = (
        provider,
        provider.settings,
        provider.kernel,
    )
    members = [
        SimpleNamespace(perf=0.003, perfs=[0.003], fn=Mock()),
        SimpleNamespace(perf=0.004, perfs=[0.004], fn=Mock()),
    ]
    isolated = Mock(return_value=[0.003, 0.004])
    monkeypatch.setattr(provider, "benchmark_isolated", isolated)
    applied = Mock()
    monkeypatch.setattr(search, "_apply_rebenchmark_timings", applied)
    monkeypatch.setattr(
        search,
        "_resolve_isolated_rebenchmark_results",
        lambda members, results: (results, [None, None]),
    )
    search.rebenchmark(members, use_isolated=False, use_interleaved=True)
    isolated.assert_called_once()
    applied.assert_called_once()
    for member in members:
        member.fn.assert_not_called()
    isolated.return_value = None
    with pytest.raises(graph.GraphScoringUnavailable, match="fall back"):
        search.rebenchmark(members)


def test_native_mirrored_refinement_and_final_windows_use_graph_jobs(
    provider, monkeypatch
):
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.benchmark_provider, search.settings, search.kernel, search.log = (
        provider,
        provider.settings,
        provider.kernel,
        provider.log,
    )
    search.args = provider.args
    members = [
        SimpleNamespace(perf=0.003, perfs=[0.003], fn=Mock()),
        SimpleNamespace(perf=0.004, perfs=[0.004], fn=Mock()),
    ]
    trace = MirroredBenchmarkTrace(
        [[0, 1], [1, 0]], [[0.003, 0.004], [0.004, 0.003]], [0.003, 0.004]
    )
    provider._benchmark_worker.run.return_value = _evidence(
        captures=2, poisoned_outputs=2, trace=trace
    )
    monkeypatch.setattr(search, "_apply_rebenchmark_timings", Mock())
    result = search.mirrored_rebenchmark(members, desc="refinement", target_ms=200)
    job = provider._benchmark_worker.run.call_args.args[0]
    assert job.repeat is not None and job.other_fn_specs == ("native-wrapper",)
    assert result.medians_ms == [0.003, 0.004]
    assert result.target_ms == 200
    policy = SimpleNamespace(warmup_ms=1, rep_ms=3, thermal_warmup_ms=1)
    assert isinstance(
        search._final_selection_measurer(members, policy), _WorkerFinalSelectionMeasurer
    )


def test_partial_wrapper_load_is_cleaned_without_graph_or_parent_fallback(
    monkeypatch, tmp_path
):
    job = _job(
        tmp_path, (torch.ones(2),), torch.ones(2), other_fn_specs=("second",), repeat=2
    )
    monkeypatch.setattr(graph, "_has_cuda_tensors", lambda arguments: True)
    first = Mock()
    loader = Mock(side_effect=[first, RuntimeError("second wrapper failed")])
    unloaded = Mock()
    monkeypatch.setattr(graph, "_load_compiled_fn_for_worker", loader)
    monkeypatch.setattr(graph, "_unload_compiled_fn", unloaded)
    with pytest.raises(RuntimeError, match="second wrapper"):
        job()
    unloaded.assert_called_once_with(first)
    first.assert_not_called()


def test_owned_graph_replays_clear_l2_before_events_and_reset_before_release(
    monkeypatch,
):
    capture = {"active": False}
    events, nodes = [], []
    owners = []

    def emit(name):
        if capture["active"]:
            nodes.append(lambda: events.append(name))
        else:
            events.append(name)

    class Stream:
        def wait_stream(self, stream):
            events.append("wait")

        def synchronize(self):
            events.append("sync")

    class Event:
        counter = 0

        def __init__(self, *, enable_timing, external):
            assert enable_timing and external
            self.name = "start" if Event.counter == 0 else "end"
            Event.counter += 1

        def record(self):
            emit(self.name)

        def elapsed_time(self, end):
            return 0.005

    class OwnedGraph:
        def replay(self):
            for node in nodes:
                node()

        def reset(self):
            assert owners[0]() is not None
            events.append("reset")

    class CacheClearer:
        def __call__(self):
            emit("clear")

    def make_clearer():
        owner = CacheClearer()
        owners.append(weakref.ref(owner))
        return owner

    @contextmanager
    def capture_graph(**kwargs):
        capture["active"] = True
        try:
            yield OwnedGraph()
        finally:
            capture["active"] = False

    @contextmanager
    def stream_context(stream):
        yield

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capture["active"]
    )
    monkeypatch.setattr(
        graph.benchmarking, "_cudagraph_unavailable_reason", lambda: None
    )
    monkeypatch.setattr(graph.benchmarking, "_make_l2_cache_clearer", make_clearer)
    monkeypatch.setattr(helion_runtime, "cute_cuda_graph", capture_graph)
    sampler = graph.GraphReplay(
        lambda: emit("kernel") or torch.ones(2),
        poison=lambda output: events.append("poison") or 1,
    )
    gc.collect()
    assert owners[0]() is not None
    events.clear()
    for _ in range(3):
        assert sampler.sample() == 0.005
    assert events == ["clear", "start", "kernel", "end", "sync"] * 3
    sampler.close()
    assert events[-2:] == ["sync", "reset"]
    assert sampler.graph is None and sampler.output is None
    gc.collect()
    assert owners[0]() is None
