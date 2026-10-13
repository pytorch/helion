from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from test import test_autotuner as fixtures

from helion.autotuner.pattern_search import PatternSearch


def exercise(scores, *, delta=0.001, initial=1.0, remeasured=1.2):
    fixture = fixtures.TestPolishDescent()
    search = fixture._make_search(polish_rounds=10)
    search.min_improvement_delta = delta
    start = fixture._member(initial, 1)
    neighbors = [fixture._member(None, i + 2) for i in range(len(scores))]
    search.population = [start]
    visited = {start.config}
    scores_by_id = {
        id(member): score for member, score in zip(neighbors, scores, strict=True)
    }
    members_by_flat = {member.flat_values: member for member in neighbors}
    generated = []

    def generate(self, flat):
        generated.append(flat)
        index = len(generated) - 1
        return [neighbors[index].flat_values] if index < len(neighbors) else []

    def benchmark(members, desc=""):
        for member in members:
            member.perfs.append(scores_by_id[id(member)])

    def rebenchmark(members, desc=""):
        members[0].perfs.append(remeasured)
        for member in members[1:]:
            member.perfs.append(scores_by_id[id(member)])

    search.benchmark_population = benchmark
    search.rebenchmark_population = rebenchmark
    with (
        patch.object(PatternSearch, "_generate_neighbors", generate),
        patch.object(search, "make_unbenchmarked", side_effect=members_by_flat.get),
    ):
        search._polish_descent(visited)
    return search, start, neighbors, generated, visited


@pytest.mark.parametrize("neighbor", [1.0, 1.01, 0.9995])
def test_noisy_remeasurement_cannot_extend_polish(neighbor):
    search, start, neighbors, generated, visited = exercise([neighbor, 0.8])
    assert start.perf == 1.2
    assert len(generated) == 1
    assert neighbors[0] in search.population and neighbors[0].config in visited
    assert neighbors[1] not in search.population
    assert neighbors[0].perfs == [neighbor, neighbor]


def test_genuine_improvement_continues_but_next_stagnation_stops():
    search, start, neighbors, generated, visited = exercise([0.9, 0.9001, 0.8])
    assert len(generated) == 2
    assert neighbors[:2] == search.population[1:]
    assert neighbors[2].config not in visited
    assert search.best is neighbors[1]


@pytest.mark.parametrize("delta, expected_rounds", [(0.02, 1), (0.001, 2), (0.0, 2)])
def test_configured_minimum_improvement_applies_to_original_score(
    delta, expected_rounds
):
    _, _, _, generated, _ = exercise([0.99], delta=delta)
    assert len(generated) == expected_rounds


def test_zero_delta_still_requires_strict_progress():
    _, _, _, generated, _ = exercise([1.0, 0.8], delta=0.0)
    assert len(generated) == 1


def test_no_finite_progress_stops():
    _, _, _, generated, _ = exercise([math.inf, 0.8], remeasured=math.inf)
    assert len(generated) == 1


def test_finite_recovery_from_infinite_incumbent_can_continue():
    _, _, _, generated, _ = exercise([1.0], initial=math.inf)
    assert len(generated) == 2


def test_same_current_object_keeps_original_stop_rule():
    _, _, _, generated, _ = exercise([1.0, 0.8], remeasured=0.8)
    assert len(generated) == 1


def test_polish_policy_version_is_serialized():
    search = fixtures.TestCuteFlashSearchPolicyCacheKey._search(effort="quick")
    policy = search._algorithm_cache_policy()
    assert policy["polish_stop_version"] == 1
    assert policy["min_improvement_delta"] == search.min_improvement_delta
    assert policy["polish_rounds"] == search.polish_rounds
    assert policy["lfbo_version"] == 3
