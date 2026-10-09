from __future__ import annotations

from contextlib import nullcontext
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from helion import Config
from helion import exc
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch


@pytest.fixture
def phase():
    return {
        "completed": True,
        "qualification_passes_completed": 10,
        "qualification_passes_planned": 10,
        "schedule_anchor_complete": True,
        "family_probe_required": True,
        "family_probe_complete": True,
        "compound_catalog_complete": True,
        "budget_exhausted": False,
        "leaf_results": [],
        "clc_families": [],
        "compound_transfers": [],
    }


@pytest.fixture
def search(monkeypatch, phase):
    member = PopulationMember(
        fn=lambda: None,
        perfs=[1.0],
        flat_values=[1],
        config=Config(block_sizes=[1, 128, 128]),
        status="ok",
    )
    result = LFBOPatternSearch.__new__(LFBOPatternSearch)
    result.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
    # Match the constructor state used by the initial-population fallback gate.
    result.best_available_pad_random = True
    result.initial_population = 1
    result.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
    result.log = Mock()
    result.log.autotune_tracing.return_value = nullcontext()
    result.copies = 1
    result.max_generations = 20
    result.similarity_penalty = 1.0
    result._generate_initial_population_flat = Mock(return_value=[[1]])
    result.make_unbenchmarked = Mock(return_value=member)
    result.set_generation = Mock()
    result.benchmark_population = Mock()
    result.compile_timeout_lower_bound = 0.0
    result.compile_timeout_quantile = 0.0
    result.set_adaptive_compile_timeout = Mock()
    result.rebenchmark_population = Mock()
    result.kernel = Mock(env=SimpleNamespace(process_group_name=None))
    result.capture_compiler_seed_members = Mock()
    result.config_gen = Mock(encode_config=lambda flat: flat)
    result._append_training_sample = Mock()
    result._fit_surrogate = Mock()
    result._autotune_metrics = AutotuneMetrics()

    def qualify(*args, **kwargs):
        result._autotune_metrics.search_phase_metrics = phase
        return 10

    result._run_flash_structural_qualification = Mock(side_effect=qualify)
    result._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
    result._select_starting_paths = Mock(return_value=[(member, ())])
    result._path_exhausts_generation_budget = Mock(return_value=False)
    result._pruned_pattern_search_from = Mock(return_value=iter(()))
    result._budgeted_range = Mock(return_value=range(0))
    result._polish_descent = Mock()
    result._finalize = Mock(return_value=member.config)
    monkeypatch.setattr(
        "helion.autotuner.surrogate_pattern_search.check_population_consistency", Mock()
    )
    return result


def incomplete_compound(phase):
    phase["completed"] = False
    phase["compound_transfers"] = [
        {
            "family": "fa4_2cta",
            "compound_packet": "deg2_16x6",
            "softmax_disc": True,
            "complete": False,
            "transfer_target_count": 2,
            "successful_transfer_config_ids": ["success"],
        }
    ]


def assert_stopped_before_search(search):
    search._select_starting_paths.assert_not_called()
    search._pruned_pattern_search_from.assert_not_called()
    search._polish_descent.assert_not_called()
    search._finalize.assert_not_called()


def test_incomplete_compound_stops_before_main_search(search, phase):
    # Family probing and every scheduled pass finished, but one transfer and
    # its bounded replacement failed: later search cannot qualify that leaf.
    incomplete_compound(phase)
    original = copy.deepcopy(phase)
    with pytest.raises(exc.AutotuneError, match="structural qualification") as error:
        search._autotune()
    assert "fa4_2cta" in str(error.value)
    assert "deg2_16x6" in str(error.value)
    assert "softmax_disc=True" in str(error.value)
    assert "successful transfers=1/2" in str(error.value)
    assert phase == original
    assert_stopped_before_search(search)


def test_incomplete_qualification_keeps_provider_cleanup_and_failure_metrics(
    search, phase
):
    incomplete_compound(phase)
    search._prepare = Mock()
    search.settings = SimpleNamespace(autotune_log=False, autotune_log_details=False)
    search.benchmark_provider = Mock()
    search._finalize_autotune_metrics = Mock()
    with pytest.raises(exc.AutotuneError, match="successful transfers=1/2"):
        search.autotune(skip_cache=True)
    search.benchmark_provider.setup.assert_called_once_with()
    search.benchmark_provider.cleanup.assert_called_once_with()
    search._finalize_autotune_metrics.assert_called_once_with(None)
    assert_stopped_before_search(search)


@pytest.mark.parametrize("metric", ["leaf_results", "clc_families"])
def test_incomplete_ordinary_or_clc_leaf_stops_search(search, phase, metric):
    phase["completed"] = False
    phase[metric] = [
        {
            "family": "fa4_clc",
            "softmax_disc": False,
            "complete": False,
        }
    ]
    if metric == "leaf_results":
        phase[metric][0]["compound_packet"] = None
    with pytest.raises(exc.AutotuneError, match="fa4_clc"):
        search._autotune()
    assert_stopped_before_search(search)


@pytest.mark.parametrize(
    "metric",
    ["schedule_anchor_complete", "family_probe_complete", "compound_catalog_complete"],
)
def test_incomplete_required_component_stops_search(search, phase, metric):
    phase["completed"] = False
    phase[metric] = False
    with pytest.raises(exc.AutotuneError, match=metric):
        search._autotune()
    assert_stopped_before_search(search)


def test_expired_wall_budget_returns_best_without_claiming_qualification(search, phase):
    incomplete_compound(phase)
    phase["budget_exhausted"] = True
    original = copy.deepcopy(phase)
    search._autotune_budget_exceeded_across_ranks.return_value = True
    assert search._autotune() == search._finalize.return_value
    search._select_starting_paths.assert_not_called()
    search._polish_descent.assert_not_called()
    search._finalize.assert_called_once_with()
    assert phase == original


@pytest.mark.parametrize("budget", [None, 10.0, 30.0])
def test_explicit_wall_budget_uses_actual_elapsed_time(
    search, phase, monkeypatch, budget
):
    incomplete_compound(phase)
    search.settings = SimpleNamespace(autotune_budget_seconds=budget)
    search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
    search._autotune_budget_start = 100.0
    search._autotune_budget_exceeded_across_ranks = (
        BaseSearch._autotune_budget_exceeded_across_ranks.__get__(search)
    )
    monkeypatch.setattr("helion.autotuner.base_search.time.perf_counter", lambda: 120.0)
    monkeypatch.setattr(
        "helion.autotuner.base_search.all_gather_object",
        lambda exceeded, **kwargs: [exceeded],
    )
    if budget == 10.0:
        assert search._autotune() == search._finalize.return_value
        search._select_starting_paths.assert_not_called()
        search._finalize.assert_called_once_with()
        assert phase["completed"] is False
    else:
        with pytest.raises(exc.AutotuneError, match="successful transfers=1/2"):
            search._autotune()
        assert_stopped_before_search(search)


def test_complete_qualification_continues_without_extra_budget_check(search, phase):
    assert search._autotune() == search._finalize.return_value
    search._select_starting_paths.assert_called_once_with()
    search._finalize.assert_called_once_with()
    search._autotune_budget_exceeded_across_ranks.assert_not_called()


def test_search_without_qualification_keeps_existing_flow(search):
    search._run_flash_structural_qualification.side_effect = None
    search._run_flash_structural_qualification.return_value = 0
    assert search._autotune() == search._finalize.return_value
    search._select_starting_paths.assert_called_once_with()
    search._autotune_budget_exceeded_across_ranks.assert_not_called()
