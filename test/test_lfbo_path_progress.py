from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from test import test_autotuner as fixtures

import helion
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

if TYPE_CHECKING:
    from collections.abc import Sequence


def exercise(
    scores: Sequence[float],
    *,
    delta: float = 0.001,
    initial: float = 1.0,
    remeasured: float = 1.2,
    patience: int = 0,
    disable: bool = False,
) -> tuple[
    PopulationMember,
    list[PopulationMember],
    list[list[PopulationMember]],
    set[helion.Config],
]:
    def member(index: int, perf: float | None) -> PopulationMember:
        return PopulationMember(
            lambda: None,
            [] if perf is None else [perf],
            (index,),
            helion.Config(num_warps=index + 1),
            status="ok",
        )

    current = member(0, initial)
    neighbors = [member(i + 1, None) for i in range(len(scores))]
    by_flat = {m.flat_values: m for m in neighbors}
    proposals = iter([[m.flat_values] for m in neighbors])
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.max_generations = len(scores) + 2
    search.patience = patience
    search.min_improvement_delta = delta
    search.frac_selected = 1.0
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.log = lambda *args, **kwargs: None
    search._generate_neighbors = lambda *args, **kwargs: next(proposals, [])
    search.make_unbenchmarked = by_flat.get
    search._surrogate_select = lambda candidates, count: candidates[:count]
    visited = {current.config}
    observed = []
    path = search._pruned_pattern_search_from(
        0, current, visited, disable_early_stopping=disable
    )
    for batch in path:
        index = len(observed)
        assert batch[1] is neighbors[index]
        batch[0].perfs.append(remeasured)
        batch[1].perfs.append(scores[index])
        observed.append(batch)
    return current, neighbors, observed, visited


@pytest.mark.parametrize("neighbor", [1.0, 1.01, 0.9995])
def test_main_path_remeasurement_does_not_create_progress(neighbor: float) -> None:
    current, neighbors, observed, visited = exercise([neighbor, 0.8])
    assert len(observed) == 1
    assert current.perfs == [1.0, 1.2]
    assert neighbors[0].perfs == [neighbor]
    assert neighbors[0].config in visited
    assert neighbors[1].perfs == []
    assert neighbors[1].config not in visited


def test_main_path_genuine_improvement_then_stagnation() -> None:
    _, neighbors, observed, visited = exercise([0.9, 0.9001, 0.8])
    assert len(observed) == 2
    assert observed[1][0] is neighbors[0]
    assert neighbors[0].perfs == [0.9, 1.2]
    assert neighbors[1].config in visited
    assert neighbors[2].config not in visited


@pytest.mark.parametrize("delta, rounds", [(0.02, 1), (0.001, 2), (0.0, 2)])
def test_main_path_preserves_configured_threshold(delta: float, rounds: int) -> None:
    _, _, observed, _ = exercise([0.99, 0.99], delta=delta)
    assert len(observed) == rounds


def test_main_path_exact_threshold_is_meaningful_progress() -> None:
    _, _, observed, _ = exercise([0.875, 0.875], delta=0.125)
    assert len(observed) == 2


def test_main_path_zero_threshold_still_requires_strict_improvement() -> None:
    _, _, observed, _ = exercise([1.0, 0.8], delta=0.0)
    assert len(observed) == 1


def test_main_path_preserves_existing_patience_allowance() -> None:
    _, neighbors, observed, visited = exercise([1.0, 1.0, 0.8], patience=1)
    assert len(observed) == 2
    assert neighbors[1].config in visited
    assert neighbors[2].config not in visited


def test_main_path_explicit_early_stop_disable_still_exhausts_candidates() -> None:
    _, neighbors, observed, visited = exercise([1.0, 1.01, 1.02], disable=True)
    assert len(observed) == 3
    assert all(m.config in visited for m in neighbors)


def test_main_path_original_same_object_stop_is_retained() -> None:
    _, _, observed, _ = exercise([1.0, 0.7], remeasured=0.8)
    assert len(observed) == 1


def test_main_path_finite_recovery_from_infinity() -> None:
    _, _, observed, _ = exercise([1.0, 0.9], initial=math.inf)
    assert len(observed) == 2


def test_main_path_no_finite_progress_stops() -> None:
    _, _, observed, _ = exercise([math.inf, 0.8], remeasured=math.inf)
    assert len(observed) == 1


def test_main_path_policy_version_is_serialized() -> None:
    search = fixtures.TestCuteFlashSearchPolicyCacheKey._search(effort="quick")
    policy = search._algorithm_cache_policy()
    assert policy["main_path_stop_version"] == 1
    assert policy["polish_stop_version"] == 1
    assert policy["min_improvement_delta"] == search.min_improvement_delta
    assert policy["patience"] == search.patience
    assert policy["lfbo_version"] == 3
