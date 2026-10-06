from __future__ import annotations

from contextlib import ExitStack
import gc
import math
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from helion._compiler.backend import TritonBackend
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.llm_seeded_lfbo import LLMSeededLFBOTreeSearch
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pipeline_proposals import _attach_objective
from helion.autotuner.pipeline_proposals import _CoordinateBudgetExhausted
from helion.autotuner.pipeline_proposals import _CoordinateHybrid
from helion.autotuner.pipeline_proposals import _KernelView
from helion.autotuner.pipeline_proposals import _not_an_isolated_stage
from helion.autotuner.pipeline_proposals import _Objective
from helion.autotuner.pipeline_proposals import _PipelineBenchmarkProvider
from helion.autotuner.pipeline_proposals import search_coordinate
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.config import Config
from helion.runtime.pipeline import PipelineBudgetExhausted
from helion.runtime.settings import Settings


def _stage() -> SimpleNamespace:
    settings = Settings(
        autotune_log=None,
        autotune_log_details=False,
        autotune_log_search_space=False,
        autotune_progress_bar=False,
    )
    return SimpleNamespace(
        config=Config(block_sizes=[32]),
        args=(),
        bound=SimpleNamespace(settings=settings),
    )


def _native_bound() -> SimpleNamespace:
    spec = ConfigSpec(backend=TritonBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
    spec.loop_orders.append(LoopOrderSpec([0]))
    return SimpleNamespace(
        settings=_stage().bound.settings,
        config_spec=spec,
        env=SimpleNamespace(process_group_name=None),
        format_kernel_decorator=lambda config, settings: "@helion.kernel",
        maybe_log_repro=lambda *args: None,
        get_cached_path=lambda config: None,
        compile_config=_not_an_isolated_stage,
        bench_compile_config=_not_an_isolated_stage,
    )


def _prepare_cpu(search: BaseSearch) -> None:
    """Replace device metadata collection, leaving the actual algorithms intact."""
    search._prepared = True
    search._autotune_metrics = AutotuneMetrics()
    search.benchmark_provider = search._benchmark_provider_cls(
        kernel=search.kernel,
        settings=search.settings,
        config_spec=search.config_spec,
        args=search.args,
        log=search.log,
        autotune_metrics=search._autotune_metrics,
    )
    search.benchmark_provider.set_budget_exceeded_fn(search._autotune_budget_exceeded)


@pytest.fixture
def overrides():
    with ExitStack() as stack:
        yield stack


def _population(objective: _Objective, overrides: ExitStack) -> PopulationBasedSearch:
    search = PopulationBasedSearch(_native_bound(), ())
    _attach_objective(search, objective, overrides)
    _prepare_cpu(search)
    return search


def _member(size: int, perf: float) -> PopulationMember:
    return PopulationMember(
        _not_an_isolated_stage, [perf], [], Config(block_sizes=[size]), "ok"
    )


def test_finite_keeps_incumbent_explicit_and_settings_seeds_without_mutation() -> None:
    stage = _stage()
    configured = Config(block_sizes=[128])
    stage.bound.settings.autotune_seed_configs = [configured]
    stage.bound.settings.autotune_effort = "quick"
    explicit = Config(block_sizes=[64])
    calls = []

    def evaluate(config: Config) -> float:
        calls.append(config.block_sizes[0])
        perf = 512 / config.block_sizes[0]
        config.block_sizes[0] = 1024
        return perf

    result = search_coordinate(
        stage,
        evaluate,
        seed_configs=[explicit, configured, explicit],
        max_evaluations=10,
        algorithm="finite",
    )
    assert calls == [32, 64, 128]
    assert [config.block_sizes for config in result.configs] == [[128], [64], [32]]
    assert result.scores == (4.0, 8.0, 16.0)
    assert result.evaluations == 3
    assert result.method == "finite"
    assert not result.budget_exhausted
    assert stage.bound.settings.autotune_effort == "quick"
    assert stage.bound.settings.autotune_seed_configs == [configured]
    assert configured.block_sizes == [128]
    assert explicit.block_sizes == [64]
    assert stage.config.block_sizes == [32]


def test_budget_retains_measured_incumbent_without_synthetic_candidate_scores() -> None:
    callback = Mock(return_value=7.0)
    result = search_coordinate(
        _stage(),
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=1,
        algorithm="finite",
    )
    assert result.configs == (Config(block_sizes=[32]),)
    assert result.scores == (7.0,)
    assert result.evaluations == callback.call_count == 1
    assert result.budget_exhausted


@pytest.mark.parametrize("invalid", [math.inf, math.nan, 0.0, -1.0])
def test_rejected_candidates_cannot_win(invalid: float) -> None:
    values = iter([2.0, invalid])
    result = search_coordinate(
        _stage(),
        lambda config: next(values),
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=3,
        algorithm="finite",
    )
    assert result.scores == (2.0,)
    assert result.evaluations == 2


def test_callback_exceptions_propagate_intact() -> None:
    error = RuntimeError("controller global budget exhausted")
    with pytest.raises(RuntimeError) as raised:
        search_coordinate(
            _stage(),
            Mock(side_effect=error),
            seed_configs=[],
            max_evaluations=1,
            algorithm="finite",
        )
    assert raised.value is error


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_never_invokes_callback(budget: object) -> None:
    callback = Mock()
    with pytest.raises((TypeError, ValueError)):
        search_coordinate(
            _stage(),
            callback,
            seed_configs=[],
            max_evaluations=budget,
            algorithm="finite",
        )
    callback.assert_not_called()


def test_boolean_objective_is_rejected() -> None:
    with pytest.raises(TypeError, match="latency"):
        search_coordinate(
            _stage(),
            lambda config: True,
            seed_configs=[],
            max_evaluations=1,
            algorithm="finite",
        )


def test_provider_counts_real_evaluations_and_has_no_stage_callable() -> None:
    settings = _stage().bound.settings
    metrics = AutotuneMetrics()
    provider = _PipelineBenchmarkProvider(
        None,
        settings,
        None,
        (),
        AutotuningLogger(settings),
        metrics,
        objective=_Objective(lambda config: 4.5, 2),
    )
    provider.setup()
    results = provider.benchmark([Config(block_sizes=[32])])
    provider.cleanup()
    assert results[0].perf == 4.5
    assert results[0].compile_time is None
    assert metrics.num_configs_tested == 1
    assert metrics.num_successful_candidate_measurements == 1
    with pytest.raises(RuntimeError, match="isolated stage"):
        results[0].fn()
    assert not provider.has_measured_source_hash("unrelated-stage-source")


def test_partial_batch_budget_records_only_actual_measurements(overrides) -> None:
    objective = _Objective(lambda config: float(config.block_sizes[0]), 2)
    search = _population(objective, overrides)
    with pytest.raises(_CoordinateBudgetExhausted):
        search.benchmark_provider.benchmark(
            [
                Config(block_sizes=[32]),
                Config(block_sizes=[64]),
                Config(block_sizes=[128]),
            ]
        )
    assert objective.evaluations == 2
    assert list(objective.latest.values()) == [32.0, 64.0]
    assert search._autotune_metrics.num_configs_tested == 2


def test_normal_and_final_rebenchmark_use_pipeline_scores(
    monkeypatch, overrides
) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    objective = _Objective(lambda config: 100 / config.block_sizes[0], 10)
    search = _population(objective, overrides)
    members = [_member(32, 0.1), _member(64, 0.2)]
    search.population = members
    search.best_perf_so_far = 0.1
    search.rebenchmark(members, use_isolated=True, use_interleaved=True)
    assert [member.perf for member in members] == [3.125, 1.5625]
    result = search.final_rebenchmark_best(members[0])
    assert result.config.block_sizes == [64]
    assert objective.evaluations == 4


def test_mirrored_rebenchmark_uses_returned_latencies_and_order(overrides) -> None:
    calls = []
    values = iter([4.0, 2.0, 6.0, 8.0])

    def evaluate(config: Config) -> float:
        calls.append(config.block_sizes[0])
        return next(values)

    objective = _Objective(evaluate, 4)
    search = _population(objective, overrides)
    members = [_member(32, 0.1), _member(64, 0.2)]
    trace = search.mirrored_rebenchmark(members, desc="test")
    assert calls == [32, 64, 64, 32]
    assert trace.orders == [[0, 1], [1, 0]]
    assert trace.elapsed_ms == [[4.0, 2.0], [6.0, 8.0]]
    assert trace.medians_ms == [6.0, 4.0]
    assert [member.perf for member in members] == [6.0, 4.0]
    assert list(objective.latest.values()) == [6.0, 4.0]


def test_failed_recheck_removes_stale_success(overrides) -> None:
    values = iter([1.0, math.inf])
    objective = _Objective(lambda config: next(values), 2)
    search = _population(objective, overrides)
    member = _member(32, objective.measure(Config(block_sizes=[32])))
    search.rebenchmark([member])
    assert member.perf == math.inf
    assert member.status == "error"
    assert objective.latest[member.config] == math.inf


def test_kernel_view_forwards_spec_but_forbids_local_compile() -> None:
    bound = _native_bound()
    settings = Settings(autotune_effort="quick")
    view = _KernelView(bound, settings)
    assert view.config_spec is bound.config_spec
    assert view.settings is settings
    assert view.settings is not bound.settings
    with pytest.raises(RuntimeError, match="isolated stage"):
        view.compile_config(Config(block_sizes=[32]))
    with pytest.raises(RuntimeError, match="isolated stage"):
        view.bench_compile_config(Config(block_sizes=[32]))


def test_real_hybrid_children_receive_shared_objective_and_lfbo_training_data(
    overrides,
) -> None:
    bound = _native_bound()
    hybrid = _CoordinateHybrid(bound, (), llm_initial_random_configs=0)
    hybrid.objective = _Objective(lambda config: config.block_sizes[0] / 100, 10)
    hybrid.seeds = (Config(block_sizes=[128]), Config(block_sizes=[64]))
    hybrid.overrides = overrides
    hybrid.objective_context = ""
    llm = hybrid._make_llm_search()
    lfbo = hybrid._make_second_stage_search(seeded=True)
    assert type(llm) is LLMGuidedSearch
    assert type(lfbo) is LFBOTreeSearch
    assert llm._build_seed_configs()[:2] == list(hybrid.seeds)
    _prepare_cpu(llm)
    _prepare_cpu(lfbo)
    llm._benchmark_and_ingest(llm._build_seed_configs(), generation=0, desc="CPU")
    hybrid._inject_seed_into_second_stage(lfbo, llm.best.config, llm)
    assert lfbo.train_y == [result.perf for result in llm._all_benchmark_results]
    assert len(lfbo.train_x) == 3
    assert lfbo._best_available_seed_configs == [llm.best.config]
    assert llm.benchmark_provider.objective is lfbo.benchmark_provider.objective


def test_native_hybrid_executes_llm_and_real_lfbo_using_only_callback(
    monkeypatch,
) -> None:
    """CPU objective plus fake transport; the LLM/LFBO search loops are real."""
    bound = _native_bound()
    stage = SimpleNamespace(
        bound=bound, config=bound.config_spec.default_config(), args=()
    )
    monkeypatch.setenv("HELION_LLM_PROVIDER", "openai_responses")
    monkeypatch.setenv("HELION_LLM_MODEL", "gpt-6-astra")
    monkeypatch.setenv("HELION_LLM_EFFORT_LEVEL", "high")
    monkeypatch.setenv("HELION_LLM_FAST_MODE", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    monkeypatch.setenv("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", "3")
    bound.settings.autotune_max_generations = 1
    profile_kwargs = LLMSeededLFBOTreeSearch.get_kwargs_from_profile

    def small_profile(profile, settings):
        kwargs = profile_kwargs(profile, settings)
        kwargs.update(llm_configs_per_round=2, llm_initial_random_configs=0)
        kwargs["second_stage_kwargs"].update(
            copies=1,
            max_generations=2,
            num_neighbors=3,
            polish_rounds=0,
            finishing_rounds=0,
        )
        return kwargs

    monkeypatch.setattr(
        LLMSeededLFBOTreeSearch, "get_kwargs_from_profile", small_profile
    )
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")
    transport = Mock(return_value='{"configs": [{"block_sizes": [64]}]}')
    monkeypatch.setattr(LLMGuidedSearch, "_call_llm", transport)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "__init__",
        Mock(side_effect=AssertionError("isolated stage provider")),
    )
    fitted = []
    original_fit = LFBOPatternSearch._fit_surrogate

    def fit(search):
        fitted.append(tuple(search.train_y))
        return original_fit(search)

    monkeypatch.setattr(LFBOPatternSearch, "_fit_surrogate", fit)
    result = search_coordinate(
        stage,
        lambda config: 1 + 32 / config.block_sizes[0],
        seed_configs=[Config(block_sizes=[128])],
        max_evaluations=100,
        effort="quick",
        objective_context='{"aggregation": "max", "stage_key": "stage_b"}',
    )
    assert transport.call_count == 1
    messages = transport.call_args.args[0]
    initial_prompt = messages[1]["content"]
    assert "complete-pipeline GPU graph replay" in initial_prompt
    assert "downstream tensor shapes" in initial_prompt
    assert '"aggregation": "max"' in initial_prompt
    assert '"stage_key": "stage_b"' in initial_prompt
    assert fitted and len(fitted[0]) >= 3
    assert result.configs
    assert result.method == "LLMSeededLFBOTreeSearch"
    assert result.hybrid_metadata["llm_model"] == "gpt-6-astra"
    assert result.hybrid_metadata["llm_effort_level"] == "high"
    assert result.hybrid_metadata["llm_fast_mode"] is True
    breakdown = result.hybrid_metadata["hybrid_stage_breakdown"]
    assert breakdown["used_llm_seed"] is True
    assert breakdown["second_stage_algorithm"] == "LFBOTreeSearch"
    assert result.hybrid_metadata["successful_llm_proposals"] == 1
    assert result.hybrid_metadata["llm_seed_handed_off"] is True
    assert result.hybrid_metadata["second_stage_started"] is True
    assert bound.settings.autotune_seed_configs is None


def test_budget_during_native_seed_stage_unwinds_and_reports_no_llm_success(
    monkeypatch,
) -> None:
    bound = _native_bound()
    original_timeout = bound.settings.autotune_compile_timeout
    original_effort = bound.settings.autotune_effort
    stage = SimpleNamespace(
        bound=bound, config=bound.config_spec.default_config(), args=()
    )
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")
    monkeypatch.setattr(
        LLMGuidedSearch, "_call_llm", lambda self, messages: '{"configs": []}'
    )
    cleanup = Mock()
    monkeypatch.setattr(_PipelineBenchmarkProvider, "cleanup", cleanup)
    callback = Mock(return_value=1.25)
    result = search_coordinate(
        stage,
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=2,
        effort="quick",
    )
    assert callback.call_count == result.evaluations == 2
    assert result.budget_exhausted
    assert result.scores == (1.25,)
    assert result.hybrid_metadata["successful_llm_proposals"] == 0
    assert result.hybrid_metadata["llm_seed_handed_off"] is False
    assert result.hybrid_metadata["second_stage_started"] is False
    assert result.hybrid_metadata["search_completed"] is False
    assert result.hybrid_metadata["hybrid_stage_breakdown"] is None
    assert cleanup.call_count == 2
    assert bound.settings.autotune_compile_timeout == original_timeout
    assert bound.settings.autotune_effort == original_effort
    assert bound.settings.autotune_seed_configs is None


def test_serialization_detaches_result_configs_and_metadata() -> None:
    result = search_coordinate(
        _stage(),
        lambda config: 1.0,
        seed_configs=[],
        max_evaluations=2,
        algorithm="finite",
    )
    serialized = result.to_dict()
    serialized["configs"][0]["block_sizes"][0] = 1024
    serialized["hybrid_metadata"]["measurements_by_phase"]["incumbent"] = 100
    assert result.configs[0].block_sizes == [32]
    assert result.hybrid_metadata["measurements_by_phase"] == {"incumbent": 1}


@pytest.mark.parametrize("exit_mode", ["normal", "coordinate_budget", "global_budget"])
def test_scoped_overrides_release_tensor_args_without_collecting_cycles(
    monkeypatch, exit_mode: str
) -> None:
    search_refs = []
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")

    def exercise_overrides(hybrid):
        search_refs.append(weakref.ref(hybrid))
        llm = hybrid._make_llm_search()
        lfbo = hybrid._make_second_stage_search(seeded=True)
        for search in (llm, lfbo):
            search_refs.append(weakref.ref(search))
            _prepare_cpu(search)
        assert llm._build_seed_configs()
        assert "complete-pipeline GPU graph replay" in llm._build_initial_prompt()
        proposal = Config(block_sizes=[64])
        llm.benchmark_provider.benchmark([proposal])
        lfbo.benchmark_provider.benchmark([proposal])
        return proposal

    monkeypatch.setattr(_CoordinateHybrid, "_autotune", exercise_overrides)

    def run_once():
        tensor = torch.zeros(3)
        tensor_ref = weakref.ref(tensor)
        stage = SimpleNamespace(
            bound=_native_bound(), config=Config(block_sizes=[32]), args=(tensor,)
        )
        calls = 0

        def evaluate(config):
            nonlocal calls
            calls += 1
            if exit_mode == "global_budget" and calls == 2:
                raise PipelineBudgetExhausted("controller budget")
            return float(tensor.numel())

        try:
            result = search_coordinate(
                stage,
                evaluate,
                seed_configs=[],
                max_evaluations=1 if exit_mode == "coordinate_budget" else 8,
            )
            assert result.budget_exhausted == (exit_mode == "coordinate_budget")
        except PipelineBudgetExhausted:
            assert exit_mode == "global_budget"
        return tensor_ref

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        tensor_ref = run_once()
        assert tensor_ref() is None
        assert len(search_refs) == 3
        assert all(search_ref() is None for search_ref in search_refs)
    finally:
        if was_enabled:
            gc.enable()


def test_override_cleanup_restores_prior_instance_methods() -> None:
    search = PopulationBasedSearch(_native_bound(), ())

    def previous(*args, **kwargs):
        return None

    search.rebenchmark = previous
    provider_factory = search._benchmark_provider_cls
    with ExitStack() as stack:
        _attach_objective(search, _Objective(lambda config: 1.0, 2), stack)
        _prepare_cpu(search)
        assert search.rebenchmark is not previous
        assert "mirrored_rebenchmark" in vars(search)
        assert "budget_exceeded_fn" in vars(search.benchmark_provider)
    assert search.rebenchmark is previous
    assert search._benchmark_provider_cls is provider_factory
    assert "mirrored_rebenchmark" not in vars(search)
    assert "_find_similar_cached_configs" not in vars(search)
    assert "budget_exceeded_fn" not in vars(search.benchmark_provider)
