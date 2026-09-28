from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.local_cache import _cute_flash_search_policy_hash
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.search_space_logger import canonical_config_id
from helion.runtime.settings import Settings


def make_search() -> Any:
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = Settings(
        autotune_log_level=logging.CRITICAL,
        autotune_progress_bar=False,
        autotune_suspicious_rebenchmark_ratio=0.99,
    )
    search.args = ()
    search.log = AutotuningLogger(search.settings)
    search.best_perf_so_far = 19.0
    search.benchmark_provider = cast(
        "Any",
        SimpleNamespace(
            mutated_arg_indices=[], benchmark_isolated=Mock(return_value=[20.0, 19.0])
        ),
    )
    search.kernel = cast(
        "Any", SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    )
    search.config_spec = cast(
        "Any",
        SimpleNamespace(
            backend_name="cute",
            cute_flash_search_enabled=False,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: f"source-{fn()}",
                get_do_bench=Mock(side_effect=AssertionError("unexpected stock timer")),
            ),
        ),
    )
    configs = [helion.Config(num_warps=4), helion.Config(num_warps=8)]
    members = [
        PopulationMember(lambda: "a", [20.0], [], configs[0], status="ok"),
        PopulationMember(lambda: "b", [19.0], [], configs[1], status="ok"),
    ]
    search._benchmarked_members = dict(zip(configs, members, strict=True))
    search._pinned_finalist_configs = {configs[0]}
    search._pinned_finalist_members = {configs[0]: members[0]}
    search.population = members
    return search


def test_final_hook_preserves_generation_worker_isolation_and_metadata() -> None:
    search = make_search()
    callback = Mock(return_value=[14.0, 15.0])
    search.settings.autotune_final_benchmark_fn = callback
    members = search.population
    search.rebenchmark(members)
    callback.assert_not_called()
    search.benchmark_provider.benchmark_isolated.assert_called_once()
    search.benchmark_provider.benchmark_isolated.reset_mock()

    def final_bench(fns, *, candidates, repeat, desc):
        assert [fn() for fn in fns] == ["a", "b"]
        assert [c["config_id"] for c in candidates] == [
            canonical_config_id(m.config) for m in members
        ]
        assert [c["config"] for c in candidates] == [dict(m.config) for m in members]
        assert [c["source_hash"] for c in candidates] == ["source-a", "source-b"]
        assert [c["pinned"] for c in candidates] == [True, False]
        assert [c["prior_perfs_ms"] for c in candidates] == [[20.0, 20.0], [19.0, 19.0]]
        assert desc == "Final verification top 2 configs"
        assert repeat >= 1
        return [14.0, 15.0]

    callback.side_effect = final_bench
    with patch.object(search, "_confirm_suspicious_rebenchmark_timings") as confirm:
        winner = search.final_rebenchmark_best(members[1])
        assert winner.config == members[0].config
        assert winner.perf == 14.0
        confirm.assert_not_called()
    callback.assert_called_once()
    search.benchmark_provider.benchmark_isolated.assert_not_called()
    # Candidate snapshots preserve the exploration population's objective.
    assert [m.perf for m in members] == [20.0, 19.0]
    assert search.settings.autotune_benchmark_fn is None


def test_default_final_uses_existing_worker(monkeypatch) -> None:
    monkeypatch.delenv("HELION_CAP_REBENCHMARK_REPEAT", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", raising=False)
    search = make_search()
    search.benchmark_provider.benchmark_isolated.return_value = [20.0, 19.0]
    assert search.final_rebenchmark_best(search.population[0]) is search.population[1]
    search.benchmark_provider.benchmark_isolated.assert_called_once()
    assert search.benchmark_provider.benchmark_isolated.call_args.kwargs["rep"] == 5000


def test_final_override_keeps_all_pinned_plus_top_remaining(monkeypatch) -> None:
    search = make_search()
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    for n in [16, 32, 64]:
        config = helion.Config(num_warps=n)
        member = PopulationMember(lambda: "extra", [float(n)], [], config, status="ok")
        search._benchmarked_members[config] = member
        search.population.append(member)
    # Two slow pinned seeds must survive alongside the two fastest remaining.
    for member in search.population[3:]:
        search._pinned_finalist_configs.add(member.config)
        search._pinned_finalist_members[member.config] = member
    seen = []

    def callback(fns, *, candidates, repeat, desc):
        seen.extend(candidates)
        return [float(i + 1) for i in range(len(fns))]

    search.settings.autotune_final_benchmark_fn = callback
    search.final_rebenchmark_best(search.population[1])
    assert len(seen) == 5  # Three pinned seeds plus two remaining candidates.
    assert sum(c["pinned"] for c in seen) == 3


def test_legacy_callback_still_controls_generation_but_final_override_wins() -> None:
    search = make_search()
    legacy = Mock(return_value=[20.0, 19.0])
    final = Mock(return_value=[14.0, 15.0])
    search.settings.autotune_benchmark_fn = legacy
    search.settings.autotune_final_benchmark_fn = final
    search.rebenchmark(search.population, confirm_suspicious=False)
    legacy.assert_called_once()
    final.assert_not_called()
    search.final_rebenchmark_best(search.population[1])
    legacy.assert_called_once()
    final.assert_called_once()
    search.benchmark_provider.benchmark_isolated.assert_not_called()


def test_final_callback_errors_propagate_without_stock_timer_fallback() -> None:
    search = make_search()
    search.settings.autotune_final_benchmark_fn = Mock(
        side_effect=RuntimeError("CUPTI")
    )
    with pytest.raises(RuntimeError, match="CUPTI"):
        search.final_rebenchmark_best(search.population[1])
    search.benchmark_provider.benchmark_isolated.assert_not_called()
    assert [m.perf for m in search.population] == [20.0, 19.0]


def test_final_callback_keeps_mutated_candidate_arguments_private() -> None:
    search = make_search()
    original = torch.tensor([0])
    search.args = (original,)
    search.benchmark_provider.mutated_arg_indices = [0]
    search.config_spec.backend.generated_source_hash = lambda fn: None

    def mutate(x):
        x.add_(1)
        return x.item()

    for member in search.population:
        member.fn = mutate

    def callback(fns, *, candidates, repeat, desc):
        assert fns[0]() == 1
        assert fns[0]() == 2
        assert fns[1]() == 1
        return [14.0, 15.0]

    search.settings.autotune_final_benchmark_fn = callback
    search.final_rebenchmark_best(search.population[1])
    assert original.item() == 0


def test_multishape_final_callback_fails_explicitly() -> None:
    search = make_search()
    search.settings.autotune_final_benchmark_fn = Mock()
    search.benchmark_provider = object.__new__(MultiShapeBenchmarkProvider)
    with pytest.raises(exc.AutotuneError, match="multi-shape"):
        search.final_rebenchmark_best(search.population[1])


def test_custom_final_objective_cannot_reuse_stock_or_other_search_cache() -> None:
    def new_search():
        search = object.__new__(RandomSearch)
        search.count = 2
        search._benchmark_provider_cls = LocalBenchmarkProvider
        search.settings = Settings()
        return search

    stock = new_search()
    assert _cute_flash_search_policy_hash(stock, cute_flash_search_enabled=False) == ""
    first, second = new_search(), new_search()
    first.settings.autotune_final_benchmark_fn = lambda fns, **kwargs: [1.0] * len(fns)
    second.settings.autotune_final_benchmark_fn = (
        first.settings.autotune_final_benchmark_fn
    )
    assert first.cache_policy() is None
    first_key = _cute_flash_search_policy_hash(first, cute_flash_search_enabled=False)
    assert first_key
    assert first_key == _cute_flash_search_policy_hash(
        first, cute_flash_search_enabled=False
    )
    assert first_key != _cute_flash_search_policy_hash(
        second, cute_flash_search_enabled=False
    )


def test_final_callback_selects_exact_unpinned_minimum(monkeypatch) -> None:
    monkeypatch.delenv(
        "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE", raising=False
    )
    search = make_search()
    search.settings.autotune_final_benchmark_fn = Mock(return_value=[14.001, 14.0])
    winner = search.final_rebenchmark_best(search.population[0])
    assert winner.config == search.population[1].config
    assert winner.perf == 14.0
