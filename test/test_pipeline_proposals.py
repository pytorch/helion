from __future__ import annotations

import gc
import math
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from helion._compiler.backend import TritonBackend
from helion.autotuner import pipeline
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import PipelineBenchmarkProvider
from helion.autotuner.benchmark_provider import _not_an_isolated_stage
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.llm_seeded_lfbo import LLMSeededLFBOTreeSearch
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pipeline_benchmark import PipelineMeasurement
from helion.autotuner.pipeline_proposals import _attach_objective
from helion.autotuner.pipeline_proposals import _CoordinateBudgetExhausted
from helion.autotuner.pipeline_proposals import _CoordinateHybrid
from helion.autotuner.pipeline_proposals import _KernelView
from helion.autotuner.pipeline_proposals import _Objective
from helion.autotuner.pipeline_proposals import _resolve_algorithm
from helion.autotuner.pipeline_proposals import search_coordinate
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.config import Config
from helion.runtime.pipeline import PipelineBudgetExhausted
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import PipelineStage
from helion.runtime.settings import Settings


def _stage() -> SimpleNamespace:
    settings = Settings(
        autotune_log=None,
        autotune_log_details=False,
        autotune_log_search_space=False,
        autotune_progress_bar=False,
        autotune_effort="quick",
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


def _population(objective: _Objective) -> PopulationBasedSearch:
    search = PopulationBasedSearch(_native_bound(), ())
    _attach_objective(search, objective)
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
    assert result.evaluations == 3
    assert result.method == "finite"
    assert not result.budget_exhausted
    assert stage.bound.settings.autotune_effort == "quick"
    assert stage.bound.settings.autotune_seed_configs == [configured]
    assert configured.block_sizes == [128]
    assert explicit.block_sizes == [64]
    assert stage.config.block_sizes == [32]


def test_budget_measures_incumbent_before_other_candidates() -> None:
    callback = Mock(return_value=7.0)
    result = search_coordinate(
        _stage(),
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=1,
        algorithm="finite",
    )
    callback.assert_called_once_with(Config(block_sizes=[32]))
    assert result.evaluations == callback.call_count == 1
    assert result.budget_exhausted


@pytest.mark.parametrize("invalid", [math.inf, math.nan, 0.0, -1.0])
def test_rejected_objective_scores_are_invalid_native_results(invalid: float) -> None:
    values = iter([2.0, invalid])
    objective = _Objective(lambda config: next(values), 3)
    search = _population(objective)
    results = search.benchmark_batch(
        [Config(block_sizes=[32]), Config(block_sizes=[64])]
    )
    assert [result.perf for result in results] == [2.0, math.inf]
    assert [result.status for result in results] == ["ok", "error"]
    assert search.best_perf_so_far == 2.0
    assert objective.evaluations == 2
    assert search._autotune_metrics.num_successful_candidate_measurements == 1


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
    provider = PipelineBenchmarkProvider(
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


def test_partial_batch_budget_records_only_actual_measurements() -> None:
    callback = Mock(side_effect=lambda config: float(config.block_sizes[0]))
    objective = _Objective(callback, 2)
    search = _population(objective)
    with pytest.raises(_CoordinateBudgetExhausted):
        search.benchmark_provider.benchmark(
            [
                Config(block_sizes=[32]),
                Config(block_sizes=[64]),
                Config(block_sizes=[128]),
            ]
        )
    assert objective.evaluations == 2
    assert [call.args[0].block_sizes for call in callback.call_args_list] == [
        [32],
        [64],
    ]
    assert search._autotune_metrics.num_configs_tested == 2


def test_normal_and_final_rebenchmark_use_pipeline_scores(monkeypatch) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    objective = _Objective(lambda config: 100 / config.block_sizes[0], 10)
    search = _population(objective)
    members = [_member(32, 0.1), _member(64, 0.2)]
    search.population = members
    search.best_perf_so_far = 0.1
    search.rebenchmark(members, use_isolated=True, use_interleaved=True)
    assert [member.perf for member in members] == [3.125, 1.5625]
    result = search.final_rebenchmark_best(members[0])
    assert result.config.block_sizes == [64]
    assert objective.evaluations == 4


def test_mirrored_rebenchmark_uses_returned_latencies_and_order() -> None:
    calls = []
    values = iter([4.0, 2.0, 6.0, 8.0])

    def evaluate(config: Config) -> float:
        calls.append(config.block_sizes[0])
        return next(values)

    objective = _Objective(evaluate, 4)
    search = _population(objective)
    members = [_member(32, 0.1), _member(64, 0.2)]
    trace = search.mirrored_rebenchmark(members, desc="test")
    assert calls == [32, 64, 64, 32]
    assert trace.orders == [[0, 1], [1, 0]]
    assert trace.elapsed_ms == [[4.0, 2.0], [6.0, 8.0]]
    assert trace.medians_ms == [6.0, 4.0]
    assert [member.perf for member in members] == [6.0, 4.0]
    assert objective.evaluations == 4


@pytest.mark.parametrize(
    "latencies, selected, perfs",
    [
        ([4.0, 2.0, 6.0, 8.0], 64, [6.0, 4.0]),
        ([4.0, 2.0, math.inf, 8.0], 32, [6.0, math.inf]),
    ],
)
def test_full_effort_final_selection_uses_pipeline_callbacks(
    monkeypatch,
    latencies: list[float],
    selected: int,
    perfs: list[float],
) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    calls = []
    values = iter(latencies)

    def evaluate(config: Config) -> float:
        calls.append(config.block_sizes[0])
        return next(values)

    objective = _Objective(evaluate, 4)
    search = _population(objective)
    search.final_selection = FinalSelectionConfig(
        top_k=2, rounds=2, thermal_warmup_ms=0, screen_rep_ms=0
    )
    members = [_member(32, 0.1), _member(64, 0.2)]
    search.population = members
    result = search.final_rebenchmark_best(members[0])
    assert calls == [32, 64, 64, 32]
    assert result.config.block_sizes == [selected]
    assert [member.perf for member in members] == perfs
    assert objective.evaluations == 4


def test_failed_recheck_removes_stale_success() -> None:
    values = iter([1.0, math.inf])
    objective = _Objective(lambda config: next(values), 2)
    search = _population(objective)
    member = _member(32, objective.measure(Config(block_sizes=[32])))
    search.rebenchmark([member])
    assert member.perf == math.inf
    assert member.status == "error"
    assert objective.evaluations == 2


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


def test_real_hybrid_children_receive_shared_objective_and_lfbo_training_data() -> None:
    bound = _native_bound()
    seeds = (Config(block_sizes=[128]), Config(block_sizes=[64]))
    bound.settings.autotune_seed_configs = seeds
    hybrid = _CoordinateHybrid(bound, (), llm_initial_random_configs=0)
    hybrid.objective = _Objective(lambda config: config.block_sizes[0] / 100, 10)
    llm = hybrid._make_llm_search()
    lfbo = hybrid._make_second_stage_search(seeded=True)
    assert isinstance(llm, LLMGuidedSearch)
    assert type(lfbo) is LFBOTreeSearch
    _prepare_cpu(llm)
    _prepare_cpu(lfbo)
    llm._benchmark_and_ingest(llm._build_seed_configs(), generation=0, desc="CPU")
    hybrid._inject_seed_into_second_stage(lfbo, llm.best.config, llm)
    assert lfbo.train_y == [result.perf for result in llm._all_benchmark_results]
    assert len(lfbo.train_x) == len(llm._all_benchmark_results)
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
        algorithm="LLMSeededLFBOTreeSearch",
        effort="quick",
    )
    assert transport.call_count == 1
    assert transport.call_args.args[0][1]["content"] == "CPU"
    assert fitted and len(fitted[0]) >= 3
    assert result.method == "LLMSeededLFBOTreeSearch"
    assert result.hybrid_metadata["llm_model"] == "gpt-6-astra"
    assert result.hybrid_metadata["llm_effort_level"] == "high"
    assert result.hybrid_metadata["llm_fast_mode"] is True
    breakdown = result.hybrid_metadata["hybrid_stage_breakdown"]
    assert breakdown["used_llm_seed"] is True
    assert breakdown["second_stage_algorithm"] == "LFBOTreeSearch"
    phases = result.hybrid_metadata["measurements_by_phase"]
    assert result.evaluations == sum(phases.values())
    assert (
        sum(count for phase, count in phases.items() if phase.endswith(":Round 0 LLM"))
        == 1
    )
    assert bound.settings.autotune_seed_configs is None


def test_budget_during_native_seed_stage_unwinds_before_llm_proposals(
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
    transport = Mock(return_value='{"configs": []}')
    monkeypatch.setattr(LLMGuidedSearch, "_call_llm", transport)
    cleanup = Mock()
    monkeypatch.setattr(PipelineBenchmarkProvider, "cleanup", cleanup)
    callback = Mock(return_value=1.25)
    result = search_coordinate(
        stage,
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=2,
        algorithm="LLMSeededLFBOTreeSearch",
        effort="quick",
    )
    assert callback.call_count == result.evaluations == 2
    assert result.budget_exhausted
    phases = result.hybrid_metadata["measurements_by_phase"]
    assert phases["incumbent"] == 1
    assert list(phases.values()) == [1, 1]
    assert next(reversed(phases)).endswith(":Round 0 seed")
    assert result.hybrid_metadata["hybrid_stage_breakdown"] is None
    assert cleanup.call_count == 2
    assert bound.settings.autotune_compile_timeout == original_timeout
    assert bound.settings.autotune_effort == original_effort
    assert bound.settings.autotune_seed_configs is None


def test_serialization_detaches_search_metadata() -> None:
    result = search_coordinate(
        _stage(),
        lambda config: 1.0,
        seed_configs=[],
        max_evaluations=2,
        algorithm="finite",
    )
    serialized = result.to_dict()
    serialized["hybrid_metadata"]["measurements_by_phase"]["incumbent"] = 100
    assert result.hybrid_metadata["measurements_by_phase"] == {"incumbent": 1}


@pytest.mark.parametrize("exit_mode", ["normal", "coordinate_budget", "global_budget"])
def test_searches_release_tensor_args_without_collecting_cycles(
    monkeypatch, exit_mode: str
) -> None:
    search_refs = []
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")

    def exercise_children(hybrid):
        search_refs.append(weakref.ref(hybrid))
        llm = hybrid._make_llm_search()
        lfbo = hybrid._make_second_stage_search(seeded=True)
        for search in (llm, lfbo):
            search_refs.append(weakref.ref(search))
            _prepare_cpu(search)
        assert llm._build_seed_configs()
        assert llm._build_initial_prompt() == "CPU"
        proposal = Config(block_sizes=[64])
        llm.benchmark_provider.benchmark([proposal])
        lfbo.benchmark_provider.benchmark([proposal])
        return proposal

    monkeypatch.setattr(_CoordinateHybrid, "_autotune", exercise_children)

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
                algorithm="LLMSeededLFBOTreeSearch",
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


def test_objective_keeps_native_methods_and_skips_stage_caches() -> None:
    search = _population(_Objective(lambda config: 1.0, 2))
    assert "rebenchmark" not in vars(search)
    assert "mirrored_rebenchmark" not in vars(search)
    assert search._find_similar_cached_configs(1, config_spec_hash="stage") == []
    assert "budget_exceeded_fn" not in vars(search.benchmark_provider)


@pytest.mark.parametrize(
    "environment, explicit, expected",
    [
        (None, None, "LFBOTreeSearch"),
        ("", None, "LFBOTreeSearch"),
        ("LFBOPatternSearch", None, "LFBOPatternSearch"),
        ("LLMGuidedSearch", None, "LLMGuidedSearch"),
        ("LLMSeededLFBOTreeSearch", None, "LLMSeededLFBOTreeSearch"),
        ("invalid", "LFBOTreeSearch", "LFBOTreeSearch"),
        ("LLMGuidedSearch", "finite", "finite"),
    ],
)
def test_algorithm_default_environment_and_explicit_override(
    monkeypatch, environment: str | None, explicit: str | None, expected: str
) -> None:
    if environment is None:
        monkeypatch.delenv("HELION_AUTOTUNER", raising=False)
    else:
        monkeypatch.setenv("HELION_AUTOTUNER", environment)
    assert _resolve_algorithm(explicit) == expected


@pytest.mark.parametrize("explicit", [None, "unsupported"])
def test_unsupported_algorithm_fails_before_measurement(monkeypatch, explicit) -> None:
    monkeypatch.setenv("HELION_AUTOTUNER", "unsupported")
    callback = Mock()
    with pytest.raises(ValueError, match="Pipeline search supports"):
        search_coordinate(
            _stage(), callback, seed_configs=[], max_evaluations=10, algorithm=explicit
        )
    callback.assert_not_called()


@pytest.mark.parametrize(
    "algorithm",
    [
        "finite",
        "LFBOTreeSearch",
        "LFBOPatternSearch",
        "LLMGuidedSearch",
        "LLMSeededLFBOTreeSearch",
    ],
)
def test_effort_none_measures_only_incumbent(monkeypatch, algorithm: str) -> None:
    callback = Mock(return_value=2.0)
    prepare = Mock(side_effect=AssertionError("disabled search"))
    monkeypatch.setattr(BaseSearch, "_prepare", prepare)
    result = search_coordinate(
        _stage(),
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=10,
        algorithm=algorithm,
        effort="none",
    )
    assert result.method == algorithm
    assert result.evaluations == 1
    callback.assert_called_once_with(Config(block_sizes=[32]))
    prepare.assert_not_called()


@pytest.mark.parametrize("algorithm", ["LFBOTreeSearch", "LFBOPatternSearch"])
def test_native_lfbo_fits_and_proposes_with_pipeline_scores(
    monkeypatch, algorithm: str
) -> None:
    """Keep the actual native profile, classifier and neighbor search."""
    bound = _native_bound()
    bound.settings.autotune_max_generations = 3
    bound.settings.autotune_random_seed = 271
    stage = SimpleNamespace(bound=bound, config=Config(block_sizes=[32]), args=())
    monkeypatch.setenv("HELION_AUTOTUNER", algorithm)
    monkeypatch.setenv("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", "5")
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "__init__",
        Mock(side_effect=AssertionError("isolated stage provider")),
    )
    cleanup = Mock()
    monkeypatch.setattr(PipelineBenchmarkProvider, "cleanup", cleanup)
    fitted = []
    original_fit = LFBOPatternSearch._fit_surrogate

    def fit(search):
        original_fit(search)
        fitted.append((type(search).__name__, tuple(search.train_y)))

    monkeypatch.setattr(LFBOPatternSearch, "_fit_surrogate", fit)
    native_cls = LFBOTreeSearch if algorithm == "LFBOTreeSearch" else LFBOPatternSearch
    original_neighbors = native_cls._generate_neighbors
    proposals = []
    learned_generations = []

    def generate_neighbors(search, *args, **kwargs):
        candidates = original_neighbors(search, *args, **kwargs)
        if search.surrogate is not None:
            proposals.extend(candidates)
            if candidates:
                learned_generations.append(search._autotune_metrics.num_generations)
        return candidates

    monkeypatch.setattr(native_cls, "_generate_neighbors", generate_neighbors)
    measured = []

    def evaluate(config: Config) -> float:
        measured.append(config.block_sizes[0])
        return 1 + 32 / config.block_sizes[0]

    result = search_coordinate(
        stage,
        evaluate,
        seed_configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
        max_evaluations=100,
        effort="quick",
    )
    assert result.method == algorithm
    assert fitted and all(name == algorithm for name, _ in fitted)
    assert all(math.isfinite(value) and value > 1 for _, ys in fitted for value in ys)
    assert proposals
    assert any(generation > 1 for generation in learned_generations)
    phases = result.hybrid_metadata["measurements_by_phase"]
    assert any("Generation " in phase for phase in phases)
    assert result.evaluations == len(measured) == sum(phases.values())
    assert measured[0] == 32 and {32, 64, 128}.issubset(measured)
    cleanup.assert_called_once()
    assert bound.settings.autotune_seed_configs is None


def test_native_llm_loop_uses_pipeline_scores_and_normal_settings(monkeypatch) -> None:
    bound = _native_bound()
    # This shared BaseSearch profile kwarg must be filtered for LLM's constructor.
    bound.settings.autotune_max_generations = 3
    stage = SimpleNamespace(bound=bound, config=Config(block_sizes=[32]), args=())
    monkeypatch.setenv("HELION_LLM_PROVIDER", "openai_responses")
    monkeypatch.setenv("HELION_LLM_MODEL", "gpt-6-astra")
    monkeypatch.setenv("HELION_LLM_EFFORT_LEVEL", "high")
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")
    profile_kwargs = LLMGuidedSearch.get_kwargs_from_profile

    def small_profile(profile, settings):
        kwargs = profile_kwargs(profile, settings)
        kwargs.update(configs_per_round=2, max_rounds=2, initial_random_configs=0)
        return kwargs

    monkeypatch.setattr(LLMGuidedSearch, "get_kwargs_from_profile", small_profile)
    transport = Mock(
        side_effect=[
            '{"configs": [{"block_sizes": [64]}]}',
            '{"configs": [{"block_sizes": [128]}]}',
        ]
    )
    monkeypatch.setattr(LLMGuidedSearch, "_call_llm", transport)
    monkeypatch.setattr(
        LocalBenchmarkProvider,
        "__init__",
        Mock(side_effect=AssertionError("isolated stage provider")),
    )
    cleanup = Mock()
    monkeypatch.setattr(PipelineBenchmarkProvider, "cleanup", cleanup)
    callback = Mock(side_effect=lambda config: 1 + 32 / config.block_sizes[0])
    result = search_coordinate(
        stage,
        callback,
        seed_configs=[],
        max_evaluations=100,
        algorithm="LLMGuidedSearch",
        effort="quick",
    )
    assert transport.call_count == 2
    assert result.method == "LLMGuidedSearch"
    assert result.hybrid_metadata["llm_model"] == "gpt-6-astra"
    assert result.hybrid_metadata["llm_provider"] == "openai_responses"
    assert result.hybrid_metadata["llm_effort_level"] == "high"
    phases = result.hybrid_metadata["measurements_by_phase"]
    assert any(phase.endswith(":Round 0 LLM") for phase in phases)
    assert any(phase.endswith(":Round 1") for phase in phases)
    assert result.evaluations == callback.call_count == sum(phases.values())
    assert {32, 64, 128}.issubset(
        call.args[0].block_sizes[0] for call in callback.call_args_list
    )
    cleanup.assert_called_once()
    assert bound.settings.autotune_seed_configs is None


@pytest.mark.parametrize(
    "algorithm", ["LFBOTreeSearch", "LFBOPatternSearch", "LLMGuidedSearch"]
)
def test_direct_native_search_budget_cleans_up_provider(monkeypatch, algorithm) -> None:
    stage = SimpleNamespace(
        bound=_native_bound(), config=Config(block_sizes=[32]), args=()
    )
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")
    monkeypatch.setattr(
        LLMGuidedSearch, "_call_llm", Mock(return_value='{"configs": []}')
    )
    cleanup = Mock()
    monkeypatch.setattr(PipelineBenchmarkProvider, "cleanup", cleanup)
    callback = Mock(return_value=1.5)
    result = search_coordinate(
        stage,
        callback,
        seed_configs=[Config(block_sizes=[64])],
        max_evaluations=2,
        algorithm=algorithm,
        effort="quick",
    )
    assert result.evaluations == callback.call_count == 2
    assert result.budget_exhausted
    cleanup.assert_called_once()
    assert stage.bound.settings.autotune_seed_configs is None


def test_llm_without_viable_native_seeds_preserves_pipeline_incumbent(
    monkeypatch,
) -> None:
    bound = _native_bound()
    incumbent = Config(block_sizes=[64])

    class Evaluator:
        def __init__(self, operation, arg_sets, **kwargs):
            self.operation = operation
            self.aggregation = kwargs["aggregation"]

        def evaluate(self, bundle):
            config = bundle.configs.get("stage", incumbent)
            valid = config == incumbent
            return PipelineMeasurement(
                bundle=PipelineConfig({"stage": config}),
                timings_ms=(1.5,) if valid else (),
                aggregate_ms=1.5 if valid else math.inf,
                traces=(("stage",),),
                stages={
                    "stage": PipelineStage(
                        "stage", SimpleNamespace(), bound, (), config, "test", []
                    )
                },
                status="ok" if valid else "error",
                error=None if valid else "candidate failed pipeline validation",
            )

    monkeypatch.setattr(pipeline, "PipelineEvaluator", Evaluator)
    monkeypatch.setattr(BaseSearch, "_prepare", _prepare_cpu)
    monkeypatch.setattr(LLMGuidedSearch, "_build_initial_prompt", lambda self: "CPU")
    profile_kwargs = LLMGuidedSearch.get_kwargs_from_profile

    def no_random_seeds(profile, settings):
        kwargs = profile_kwargs(profile, settings)
        kwargs["initial_random_configs"] = 0
        return kwargs

    monkeypatch.setattr(LLMGuidedSearch, "get_kwargs_from_profile", no_random_seeds)
    transport = Mock(return_value='{"configs": []}')
    monkeypatch.setattr(LLMGuidedSearch, "_call_llm", transport)
    cleanup = Mock()
    monkeypatch.setattr(PipelineBenchmarkProvider, "cleanup", cleanup)
    result = pipeline.autotune_pipeline(
        lambda x: x,
        [(1,)],
        reference=lambda x: x,
        check=lambda actual, expected: None,
        algorithm="LLMGuidedSearch",
        effort="quick",
        rounds=1,
        beam_width=1,
    )
    assert result.config.configs["stage"] == incumbent
    assert result.aggregate_ms == result.incumbent_aggregate_ms == 1.5
    assert any(entry["status"] == "error" for entry in result.history)
    assert result.metadata["coordinate_searches"][0]["evaluations"] == 2
    transport.assert_called_once()
    cleanup.assert_called_once()
