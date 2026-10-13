from __future__ import annotations

import dataclasses
import hashlib
import inspect
import logging
import math
import pickle
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion.autotuner import benchmark_provider
from helion.autotuner import local_cache
from helion.autotuner import objective_timing
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_objective import BenchmarkObjectiveError
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.objective_timing import ObjectiveTiming
from helion.autotuner.pattern_search import PatternSearch
from helion.autotuner.precompile_future import PrecompileFuture
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.config import Config
from helion.runtime.settings import Settings


class Factory:
    def __init__(self):
        raw = torch.arange(12, dtype=torch.float32)
        self.raw_args = (raw[:4], raw[4:8], raw[8:12])
        self.record = {
            "partner": "fixed-source/config",
            "raw": "float32-alias-12",
            "oracle": "exact",
            "timer": "fixture",
        }
        self.events = []
        self.times = {1: 8.0, 2: 3.0, 3: 6.0}
        self.bad = set()
        self.late_bad = set()
        self.private = []

    def cache_record(self):
        return self.record

    def bind(self, request, private_args):
        self.private.append(private_args)
        x, state, out = private_args
        assert (
            x.untyped_storage()._cdata
            == state.untyped_storage()._cdata
            == out.untyped_storage()._cdata
        )
        assert x.data_ptr() != self.raw_args[0].data_ptr()
        rank = request.config.config["num_warps"]
        self.events.append(("bind", rank, request.source_hash))
        calls = 0

        def reset():
            self.events.append(("reset", rank))
            state.fill_(4)
            out.fill_(-9)

        def run():
            nonlocal calls
            assert torch.equal(state, torch.full_like(state, 4))
            calls += 1
            self.events.append(("run", rank))
            prepared = x * 2  # incumbent preparation inside the measured call
            request.compiled(prepared, state, out)
            if rank in self.bad or (rank in self.late_bad and calls > 1):
                out.add_(1)
            return out, rank

        def validate(output):
            self.events.append(("validate", rank))
            assert torch.equal(output[0], x * 2 + 4)
            assert torch.equal(x, torch.arange(4.0))

        def close():
            self.events.append(("close", rank))

        return BenchmarkWorkload(run, reset, validate, close)


def stage(prepared, state, out):
    out.copy_(prepared + state)
    state.add_(1)


def make(
    monkeypatch,
    cls=LFBOTreeSearch,
    *,
    factory=None,
    objective=True,
    settings_changes=None,
):
    factory = factory or Factory()
    settings = Settings(
        autotune_benchmark_subprocess=False,
        autotune_precompile=None,
        autotune_progress_bar=False,
        autotune_log_level=logging.CRITICAL,
        **(settings_changes or {}),
    )
    backend = Mock(name="backend")
    backend.name = "cute"
    backend.config_value_priors_version = 1
    backend.generated_source_hash.side_effect = lambda fn: hashlib.sha256(
        inspect.getsource(fn).encode()
    ).hexdigest()
    backend.get_do_bench.return_value = None
    backend.get_paired_device_micros_bench.return_value = None
    backend.should_deduplicate_generated_sources.return_value = True
    backend.autotune_config_is_viable.return_value = True
    gen = Mock()
    gen.flatten.side_effect = lambda c: (c.config["num_warps"],)
    gen.invalid_config_count = 0
    gen.flash_population_floor.side_effect = lambda value: value
    spec = Mock()
    spec.backend = backend
    spec.backend_name = "cute"
    spec.cute_flash_search_enabled = False
    spec.compiler_seed_timeout_retry_repetitions = None
    spec.compiler_coverage_groups = ()
    spec.compiler_seed_configs = ()
    spec.compiler_seed_witnesses = ()
    spec.create_config_generation.return_value = gen
    kernel = Mock()
    kernel.settings = settings
    kernel.config_spec = spec
    kernel.env.device = torch.device("cuda:0")  # metadata only
    kernel.env.process_group_name = None
    kernel.compile_config.return_value = stage
    kernel.extra_cache_key.return_value = "original-extra"
    kernel.configs = [Config(num_warps=1), Config(num_warps=2)]
    kernel.env.backend = backend
    kwargs = (
        {
            "benchmark_objective": factory,
            "objective_timing": ObjectiveTiming(
                lambda: factory.events.append(("flush",)),
                "required",
                "fixture-flush+cuda0",
            ),
        }
        if objective
        else {}
    )
    search = cls(kernel, (), **kwargs)
    search._autotune_metrics = AutotuneMetrics()
    # Exercise the actual provider's compile/precompile/benchmark loop. Only the
    # device precompiler and the CUDA event primitive are CPU stand-ins.
    monkeypatch.setattr(
        LocalBenchmarkProvider, "_compute_baseline", lambda s: (None, [], None)
    )
    provider = LocalBenchmarkProvider(
        kernel,
        settings,
        spec,
        (),
        search.log,
        search._autotune_metrics,
        objective_benchmark=search.objective_benchmark,
    )
    search.benchmark_provider = provider
    provider.set_budget_exceeded_fn(lambda: False)
    search._autotune_budget_exceeded_across_ranks = lambda: False
    search._budgeted_range = lambda *a: iter(range(*a))
    calls = []

    def timer(workloads, **kw):
        calls.append(kw)
        results = []
        for w in workloads:
            samples = []
            for _ in range(2):
                w.reset()
                kw["clear_cache"]()
                output, rank = w.run()
                samples.append(factory.times[rank])
            results.append(SimpleNamespace(samples_ms=tuple(samples)))
        return tuple(results)

    monkeypatch.setattr(objective_timing, "benchmark_workloads", timer)
    monkeypatch.setattr(benchmark_provider, "synchronize_device", lambda: None)
    monkeypatch.setattr(
        provider,
        "_create_precompile_future",
        lambda config, fn: PrecompileFuture.skip(
            provider._precompile_context(), config, True
        ),
    )
    return search, factory, calls


def member(rank):
    return PopulationMember(lambda: None, [], (rank,), Config(num_warps=rank))


def test_real_initial_repeat_polish_final_routes(monkeypatch):
    search, factory, calls = make(monkeypatch)
    first = member(1)
    search.benchmark_initial_population([first])
    assert first.fn is stage and first.perf == 8
    second = member(2)
    search.benchmark_population([second])
    search.rebenchmark([first, second], desc="repeat", target_ms=27)
    search.population = [first]
    third = member(3)
    second = member(2)
    search.polish_rounds = 2
    neighbors = {(1,): [(2,)], (2,): [(3,)]}
    monkeypatch.setattr(
        PatternSearch, "_generate_neighbors", lambda self, flat: neighbors[flat]
    )
    monkeypatch.setattr(
        search, "make_unbenchmarked", lambda flat: {(2,): second, (3,): third}[flat]
    )
    monkeypatch.setattr(search, "should_rebenchmark", lambda member: True)
    search._polish_descent({first.config})
    assert search.best.config == Config(num_warps=2)
    chosen = search.final_rebenchmark_best(search.best)
    assert chosen.fn is stage and chosen.config == Config(num_warps=2)
    phases = [p for p, _ in search.objective_benchmark.observations]
    assert "repeat" in phases
    assert "Polish round 1: verifying" in phases
    assert "Polish round 2: verifying" in phases
    assert any(p.startswith("Final verification") for p in phases)
    assert calls[0]["rep_ms"] == 50
    assert any(c["rep_ms"] == 27 for c in calls)
    assert all(c["warmup_ms"] == 1 and c["graph_policy"] == "required" for c in calls)
    assert all(
        r.measurement.epoch == search.objective_benchmark.epoch
        for _, r in search.objective_benchmark.observations
    )
    assert len(factory.private) == len(
        {a[0].untyped_storage()._cdata for a in factory.private}
    )
    assert torch.equal(factory.raw_args[1], torch.arange(4.0, 8.0))


def test_finite_actual_search_body_and_no_single_stage_timer(monkeypatch):
    search, factory, calls = make(monkeypatch, FiniteSearch)
    with patch.object(
        benchmark_provider, "do_bench", side_effect=AssertionError("stage timer")
    ):
        result = search._autotune()
    assert result == Config(num_warps=2)
    search.benchmark_provider.check_objective_selection(result)
    assert len(calls) == 2


@pytest.mark.parametrize("late", [False, True])
def test_bad_full_boundary_candidate_rejected_before_selection(monkeypatch, late):
    factory = Factory()
    (factory.late_bad if late else factory.bad).add(2)
    search, factory, calls = make(monkeypatch, FiniteSearch, factory=factory)
    results = search.benchmark_batch(search.configs)
    assert [r.status for r in results] == ["ok", "accuracy_error"]
    assert results[1].objective_measurement is None
    assert search.benchmark_provider._autotune_metrics.num_accuracy_failures == 1
    assert factory.events.count(("close", 2)) == 1
    with pytest.raises(BenchmarkObjectiveError):
        search.benchmark_provider.check_objective_selection(Config(num_warps=2))


def test_failed_recheck_invalidates_history(monkeypatch):
    search, factory, calls = make(monkeypatch)
    a, b = member(1), member(2)
    search.benchmark_population([a, b])
    search.population = [a, b]
    factory.late_bad.add(2)
    search.rebenchmark([a, b])
    assert math.isinf(b.perf)
    assert search.final_rebenchmark_best(a).config == a.config
    assert b.config not in search._benchmarked_members


@pytest.mark.parametrize(
    "route",
    ["initial", "repeat", "final", "history", "cache", "provider", "get", "put"],
)
def test_stale_epoch_rejected_before_any_work(monkeypatch, tmp_path, route):
    search, factory, calls = make(monkeypatch)
    factory.record["partner"] = "replaced"
    cache = object.__new__(local_cache.LocalAutotuneCache)
    cache.autotuner = search
    cache._get_local_cache_path = lambda: tmp_path / "must-not-read"
    with pytest.raises(BenchmarkObjectiveError, match="changed"):
        {
            "initial": lambda: search.benchmark_batch([]),
            "repeat": lambda: search.rebenchmark([]),
            "final": lambda: search.final_rebenchmark_best(member(1)),
            "history": lambda: search._record_benchmarked_member(member(1)),
            "cache": lambda: search._objective_cache_key("x"),
            "provider": lambda: search.benchmark_provider.benchmark([]),
            "get": cache.get,
            "put": lambda: cache.put(Config()),
        }[route]()
    assert not calls and not factory.events


@pytest.mark.parametrize("route", ["search", "provider"])
def test_dropped_objective_rejects(monkeypatch, route):
    search, factory, calls = make(monkeypatch)
    owner = search if route == "search" else search.benchmark_provider
    owner.objective_benchmark = None
    with pytest.raises(BenchmarkObjectiveError, match="removed"):
        owner._validate_timing_policy()


@pytest.mark.parametrize(
    "option", ["isolated", "confirmation", "mirrored", "serialize", "provider"]
)
def test_unsupported_routes_decline(monkeypatch, option):
    search, factory, calls = make(monkeypatch)
    with pytest.raises(BenchmarkObjectiveError):
        if option == "isolated":
            search.rebenchmark([], use_isolated=True)
        elif option == "confirmation":
            search.rebenchmark([], confirm_suspicious=True)
        elif option == "mirrored":
            search.mirrored_rebenchmark([], desc="mirrored")
        elif option == "serialize":
            pickle.dumps(search)
        else:
            search.benchmark_provider.benchmark_isolated([], warmup=1, rep=1)
    assert not calls


@pytest.mark.parametrize(
    "setting",
    [
        "subprocess",
        "profiler",
        "custom_timer",
        "custom_oracle",
        "no_accuracy",
        "remote",
        "isolated_env",
        "backend_timer",
        "flash",
    ],
)
def test_reject_incompatible_policy_before_provider(monkeypatch, setting):
    search, factory, calls = make(monkeypatch)
    if setting == "subprocess":
        search.settings.autotune_benchmark_subprocess = True
    elif setting == "profiler":
        search.settings.autotune_timing_method = "torch_profiler"
    elif setting == "custom_timer":
        search.settings.autotune_benchmark_fn = lambda fn: 1
    elif setting == "custom_oracle":
        search.settings.autotune_baseline_fn = lambda: None
    elif setting == "no_accuracy":
        search.settings.autotune_accuracy_check = False
    elif setting == "remote":
        search.settings.autotune_cache = "RemoteAutotuneCache"
    elif setting == "isolated_env":
        monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED", "1")
    elif setting == "backend_timer":
        search.config_spec.backend.get_do_bench.return_value = lambda: None
    elif setting == "flash":
        search.config_spec.cute_flash_search_enabled = True
    with pytest.raises(BenchmarkObjectiveError):
        search.benchmark_batch([])
    assert not calls


@pytest.mark.parametrize(
    "corruption",
    ["epoch", "request", "sample", "untagged", "score", "config", "source"],
)
def test_population_evidence_rejects(monkeypatch, corruption):
    search, factory, calls = make(monkeypatch)
    a = member(1)
    search.benchmark_population([a])
    if corruption == "epoch":
        a.objective_measurement = dataclasses.replace(
            a.objective_measurement, epoch="foreign"
        )
    elif corruption == "request":
        a.objective_measurement = dataclasses.replace(
            a.objective_measurement, request_digest="foreign"
        )
    elif corruption == "sample":
        a.objective_measurement = dataclasses.replace(
            a.objective_measurement, samples_ms=(True,)
        )
    elif corruption == "untagged":
        a.objective_measurement = None
    elif corruption == "score":
        a.perfs[-1] = 1.23
    elif corruption == "config":
        a.config = Config(num_warps=2)
    else:
        search.config_spec.backend.generated_source_hash.return_value = None
        search.config_spec.backend.generated_source_hash.side_effect = None
    with pytest.raises(BenchmarkObjectiveError):
        search._record_benchmarked_member(a)


def test_source_equal_configs_not_deduplicated(monkeypatch):
    search, factory, calls = make(monkeypatch, FiniteSearch)
    results = search.benchmark_batch(search.configs)
    assert len(calls) == 2
    assert [r.perf for r in results] == [8, 3]
    assert all(r.status == "ok" for r in results)
    assert results[0].fn is results[1].fn
    assert (
        results[0].objective_measurement.request_digest
        != results[1].objective_measurement.request_digest
    )


@pytest.mark.parametrize("mode", ["fork", "spawn"])
def test_candidate_precompile_context_and_routes_are_preserved(monkeypatch, mode):
    search, factory, calls = make(monkeypatch)
    provider = search.benchmark_provider
    provider.settings.autotune_precompile = mode
    provider._precompile_args_path = "stage-args-only.pt"
    ctx = provider._precompile_context()
    assert set(vars(ctx)) == {"settings", "log", "kernel", "args", "jobs"}
    with patch.object(PrecompileFuture, "create", return_value="future") as create:
        future = LocalBenchmarkProvider._create_precompile_future(
            provider, Config(), stage
        )
    assert future == "future"
    assert create.call_args.kwargs["fn"] is stage
    assert create.call_args.kwargs["ctx"].args == ()
    assert not calls and not factory.events
    provider.cleanup()


def test_local_cache_epoch_and_default_key(monkeypatch):
    search, factory, calls = make(monkeypatch)

    def key(s):
        cache = object.__new__(local_cache.LocalAutotuneCache)
        cache.autotuner = s
        cache.kernel = s.kernel
        cache.args = ()
        cache.kernel.kernel.fn = test_local_cache_epoch_and_default_key
        cache.kernel.kernel._create_bound_kernel_cache_key.return_value = (
            SimpleNamespace(
                specialization_key=(32,), extra_results=(), compiler_seed_results=()
            )
        )
        cache.kernel.config_spec.cache_fingerprint_hash.return_value = "spec"
        return cache._generate_key()

    monkeypatch.setattr(
        local_cache, "extract_device", lambda args: torch.device("cuda")
    )
    monkeypatch.setattr(local_cache, "get_device_name", lambda dev: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # no native call
    first = key(search)
    second, _, _ = make(monkeypatch, factory=factory)
    assert first == key(second)
    newfactory = Factory()
    newfactory.record["partner"] = "other"
    other, _, _ = make(monkeypatch, factory=newfactory)
    assert first != key(other)
    default, _, _ = make(monkeypatch, objective=False)
    dkey = key(default)
    assert dkey.extra_cache_key == "original-extra"
    assert (
        dataclasses.replace(
            first,
            extra_cache_key=dkey.extra_cache_key,
            search_policy_hash=dkey.search_policy_hash,
        )
        == dkey
    )


def test_timing_policy_separates_cache_epoch(monkeypatch):
    a, f, _ = make(monkeypatch)
    b, _, _ = make(monkeypatch)
    assert a.objective_benchmark.epoch == b.objective_benchmark.epoch
    object.__setattr__(a.objective_benchmark._timing, "graph_policy", "disabled")
    with pytest.raises(BenchmarkObjectiveError, match="changed"):
        a._objective_cache_key("x")


def test_timing_exception_propagates_and_closes(monkeypatch):
    search, factory, _ = make(monkeypatch)
    monkeypatch.setattr(
        objective_timing,
        "benchmark_workloads",
        Mock(side_effect=RuntimeError("capture failed")),
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        search.benchmark_batch([Config(num_warps=1)])
    assert factory.events[-1] == ("close", 1)


@pytest.mark.parametrize("which", ["trial", "rebench"])
@pytest.mark.parametrize("change", ["drop", "foreign"])
def test_outer_cache_rejects_changed_factory_epoch(monkeypatch, which, change):
    from helion.autotuner.base_cache import AutotuneCacheBase

    search, _, _ = make(monkeypatch)
    f = Factory()
    f.record["partner"] = "other"
    other, _, _ = make(monkeypatch, factory=f, objective=change != "drop")
    cache = object.__new__(local_cache.LocalAutotuneCache)
    cache.autotuner = search
    cache._autotuner_factory = lambda: other
    with pytest.raises(BenchmarkObjectiveError, match="epoch"):
        if which == "trial":
            AutotuneCacheBase._run_one_trial(cache, 0, 2, 0, skip_cache=True)
        else:
            cache._run_rebench_round([Config(num_warps=1)])


def test_outer_cache_rebench_uses_same_workload(monkeypatch):
    search, factory, _ = make(monkeypatch)
    other, _, calls = make(monkeypatch, factory=factory)
    monkeypatch.setattr(other, "_prepare", lambda: None)
    cache = object.__new__(local_cache.LocalAutotuneCache)
    cache.autotuner = search
    cache._autotuner_factory = lambda: other
    results = cache._run_rebench_round([Config(num_warps=1), Config(num_warps=2)])
    assert results == [8, 3]
    assert len(calls) == 2
    assert [p for p, _ in other.objective_benchmark.observations] == [
        "Best-of-K rebench"
    ] * 2


def test_cleanup_keeps_compiled_fn_identity_and_releases_fast_path(monkeypatch):
    search, factory, calls = make(monkeypatch)
    cleared = []
    monkeypatch.setattr(
        search.benchmark_provider, "_clear_jit_fast_path_caches", cleared.append
    )
    search.benchmark_batch([Config(num_warps=1)])
    assert cleared == [stage]
    monkeypatch.setattr(
        objective_timing, "benchmark_workloads", Mock(side_effect=RuntimeError("timer"))
    )
    with pytest.raises(RuntimeError):
        search.benchmark_batch([Config(num_warps=1)])
    assert cleared == [stage, stage]
    assert factory.events[-1] == ("close", 1)


def test_constructor_rejects_unsupported_before_raw_clone(monkeypatch):
    search, factory, calls = make(monkeypatch)
    search.settings.autotune_benchmark_subprocess = True
    with pytest.raises(BenchmarkObjectiveError):
        FiniteSearch(
            search.kernel,
            (),
            benchmark_objective=factory,
            objective_timing=ObjectiveTiming(lambda: None, "required", "fixture"),
        )
    assert not factory.events and not calls


def test_provider_construction_skips_stage_only_oracle(monkeypatch):
    search, factory, calls = make(monkeypatch)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "_compute_baseline",
        Mock(side_effect=AssertionError("stage-only oracle")),
    )
    p = LocalBenchmarkProvider(
        search.kernel,
        search.settings,
        search.config_spec,
        (),
        search.log,
        AutotuneMetrics(),
        objective_benchmark=search.objective_benchmark,
    )
    assert p.mutated_arg_indices == []
    assert not factory.events


def test_no_objective_default_stage_timing_is_retained(monkeypatch):
    search, _, _ = make(monkeypatch, FiniteSearch, objective=False)
    p = search.benchmark_provider

    def compiled(*args):
        return None

    search.kernel.compile_config.return_value = compiled
    search.kernel.bench_compile_config.side_effect = lambda *a: lambda: "ordinary"
    monkeypatch.setattr(p, "_check_accuracy", lambda *args: None, raising=False)
    monkeypatch.setattr(benchmark_provider, "do_bench", lambda fn, **kw: 7.0)
    p._baseline_output = None
    results = p.benchmark([Config(num_warps=1)])
    assert results[0].objective_measurement is None
    assert search.objective_benchmark is None


def test_timer_body_byte_exact_to_sealed95():
    from pathlib import Path

    from helion.autotuner import workload_timer

    path = Path(workload_timer.__file__)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b56cc71c0a5ad52c5451530e994dd95d36e87b97340e429fd77dc71d1312d05e"
    )


@pytest.mark.parametrize(
    "corruption",
    ["missing", "epoch", "config", "source", "score", "nan_sample", "bool_sample"],
)
def test_external_training_rejects_foreign_or_missing_evidence(monkeypatch, corruption):
    search, factory, _ = make(monkeypatch)
    result = search.benchmark_batch([Config(num_warps=1)])[0]
    measurement = result.objective_measurement
    if corruption == "missing":
        result = result._replace(objective_measurement=None)
    elif corruption == "epoch":
        result = result._replace(
            objective_measurement=dataclasses.replace(measurement, epoch="foreign")
        )
    elif corruption == "config":
        result = result._replace(config=Config(num_warps=2))
    elif corruption == "source":
        search.config_spec.backend.generated_source_hash.side_effect = lambda fn: (
            "f" * 64
        )
    elif corruption == "score":
        result = result._replace(perf=result.perf + 1)
    elif corruption == "nan_sample":
        result = result._replace(
            objective_measurement=dataclasses.replace(
                measurement, samples_ms=(math.nan,)
            )
        )
    elif corruption == "bool_sample":
        result = result._replace(
            objective_measurement=dataclasses.replace(measurement, samples_ms=(True,))
        )
    with pytest.raises(BenchmarkObjectiveError):
        search.seed_training_data([result])
    assert not search.train_x and not search.train_y


def test_external_training_accepts_same_objective_and_rejects_stale_fit(monkeypatch):
    search, factory, _ = make(monkeypatch)
    result = search.benchmark_batch([Config(num_warps=1)])[0]
    search.seed_training_data([result])
    assert search.train_y == [8]
    factory.record["partner"] = "changed"
    with pytest.raises(BenchmarkObjectiveError, match="changed"):
        search._sync_training_labels()


def test_source_alias_score_repairs_are_explicitly_declined(monkeypatch):
    search, _, _ = make(monkeypatch)
    result = search.benchmark_batch([Config(num_warps=1)])[0]
    with pytest.raises(BenchmarkObjectiveError, match="Source-only"):
        search._apply_effective_source_repairs({result.config: result}, [])


def test_direct_finite_training_append_requires_evidence(monkeypatch):
    search, _, _ = make(monkeypatch)
    with pytest.raises(BenchmarkObjectiveError):
        search._append_training_sample((1,), 1.0, Config(num_warps=1), stage)


def test_failed_recheck_invalidates_external_rows_by_exact_config(monkeypatch):
    search, factory, _ = make(monkeypatch)
    a, b = member(1), member(2)
    search.benchmark_population([a, b])
    results = search.benchmark_batch([a.config, b.config])
    assert results[0].fn is results[1].fn
    search.seed_training_data(results)
    assert search.train_configs is None
    assert search._train_members == [None, None]
    assert search.train_y == [8, 3]
    factory.late_bad.add(2)
    search.rebenchmark([a, b])
    assert math.isinf(b.perf)
    assert search.train_y == [8, math.inf]
    search._sync_training_labels()
    assert search.train_y == [8, math.inf]


def test_external_training_config_identity_is_snapshotted(monkeypatch):
    search, factory, _ = make(monkeypatch)
    result = search.benchmark_batch([Config(num_warps=2)])[0]
    search.seed_training_data([result])
    result.config.config["num_warps"] = 1
    search._invalidate_rebenchmark_training_targets({Config(num_warps=2)}, set())
    assert search.train_y == [math.inf]
