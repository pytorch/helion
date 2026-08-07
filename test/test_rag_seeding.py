"""CPU-only tests for the LFBO seeded-population builder and attempt budget.

These exercise ``helion.autotuner.rag.seeding`` with fakes: configs are strings
of the form ``"<key>:<id>"`` so distinct raw strings can normalize to the same
dedup key, and ``"bad:*"`` marks an invalid config. No autotuner, CUDA, or
retrieval dependency is needed.
"""

from __future__ import annotations

import pytest

from helion.autotuner.rag.seeding import AttemptBudget
from helion.autotuner.rag.seeding import AttemptCategory
from helion.autotuner.rag.seeding import PopulationUnderfilled
from helion.autotuner.rag.seeding import SeedingResult
from helion.autotuner.rag.seeding import build_seeded_population


def _key(config: str) -> str:
    """Normalized dedup key: the prefix before ``:`` (raw ids collapse)."""
    return config.split(":")[0]


def _is_valid(config: str) -> bool:
    return not config.startswith("bad")


def _strict_key(config: str) -> str:
    """Like ``_key`` but raises if handed an invalid config.

    Mirrors the live path where the dedup key comes from a flatten/unflatten
    round-trip that raises ``InvalidConfig`` on a bad config, proving the builder
    validates before it keys.
    """
    if not _is_valid(config):
        raise AssertionError("flatten_key called on an invalid config")
    return _key(config)


def _sequence_drawer(values):
    """A ``draw_random`` that yields ``values`` in order (then raises)."""
    it = iter(values)

    def draw() -> str:
        return next(it)

    return draw


def _counting_valid(seen_calls: list[str]):
    def is_valid(config: str) -> bool:
        seen_calls.append(config)
        return _is_valid(config)

    return is_valid


def _big_budget() -> AttemptBudget:
    return AttemptBudget(10_000)


def test_seeds_deduped_by_flatten_key_and_invalid_dropped() -> None:
    budget = _big_budget()
    # "a:1"/"a:2" collapse to key "a"; "bad:1" is invalid. flatten_key must never
    # see "bad:1" (validation happens first), enforced by _strict_key.
    population, result = build_seeded_population(
        ["a:1", "a:2", "bad:1", "b:1"],
        n=2,
        num_neighbors_cap=10,
        draw_random=_sequence_drawer([]),
        is_valid=_is_valid,
        flatten_key=_strict_key,
        budget=budget,
        draw_cap=10,
    )
    assert population == ["a:1", "b:1"]
    assert result == SeedingResult(
        attempted=4, valid=2, duplicates=1, invalid=1, random_draws=0
    )
    assert budget.spent_by(AttemptCategory.INITIAL_POPULATION) == 2
    assert budget.spent_by(AttemptCategory.DUPLICATE) == 1
    assert budget.spent_by(AttemptCategory.INVALID) == 1


def test_seeds_capped_to_num_neighbors_cap_before_construction() -> None:
    budget = _big_budget()
    seen_calls: list[str] = []
    # cap=2 admits only "a:1","b:1"; "c:1" is dropped by the cap. n=3 then needs
    # one random draw. If the cap were not applied, "c:1" would fill N and no draw
    # would happen.
    population, result = build_seeded_population(
        ["a:1", "b:1", "c:1"],
        n=3,
        num_neighbors_cap=2,
        draw_random=_sequence_drawer(["r:1"]),
        is_valid=_counting_valid(seen_calls),
        flatten_key=_key,
        budget=budget,
        draw_cap=10,
    )
    assert population == ["a:1", "b:1", "r:1"]
    assert result.random_draws == 1
    assert "c:1" not in seen_calls  # capped before construction, never considered


def test_draws_until_exactly_n_unique_valid() -> None:
    budget = _big_budget()
    # Draw stream mixes a new valid, an invalid, a duplicate, then two new valids.
    population, result = build_seeded_population(
        [],
        n=3,
        num_neighbors_cap=8,
        draw_random=_sequence_drawer(["a:1", "bad:1", "a:2", "b:1", "c:1"]),
        is_valid=_is_valid,
        flatten_key=_key,
        budget=budget,
        draw_cap=10,
    )
    assert population == ["a:1", "b:1", "c:1"]
    assert result == SeedingResult(
        attempted=5, valid=3, duplicates=1, invalid=1, random_draws=5
    )


def test_draw_cap_reached_before_n_raises_and_does_not_shrink() -> None:
    budget = _big_budget()
    with pytest.raises(PopulationUnderfilled) as excinfo:
        build_seeded_population(
            [],
            n=3,
            num_neighbors_cap=8,
            draw_random=_sequence_drawer([f"bad:{i}" for i in range(4)]),
            is_valid=_is_valid,
            flatten_key=_key,
            budget=budget,
            draw_cap=4,
        )
    err = excinfo.value
    assert err.requested == 3
    assert err.realized == 0
    assert err.draw_cap == 4
    assert err.budget_exhausted is False
    # every draw was still charged (arm failed, budget not silently reset)
    assert budget.spent() == 4


def test_attempt_budget_accounts_every_category_and_limit_behavior() -> None:
    budget = AttemptBudget(3)
    assert budget.limit == 3
    assert budget.spent() == 0
    assert budget.remaining() == 3
    assert budget.exhausted is False

    budget.record(AttemptCategory.INITIAL_POPULATION)
    assert budget.remaining() == 2
    assert budget.exhausted is False

    budget.record(AttemptCategory.INVALID)
    budget.record(AttemptCategory.DUPLICATE)
    assert budget.spent() == 3
    assert budget.remaining() == 0
    assert budget.exhausted is True

    # Later phases use the same frozen total and cannot overshoot it.
    assert not budget.record(AttemptCategory.GENERATION, n=2)
    assert not budget.record(AttemptCategory.LLM_PROPOSED)
    assert budget.spent() == 3
    assert budget.by_category() == {
        AttemptCategory.INITIAL_POPULATION: 1,
        AttemptCategory.INVALID: 1,
        AttemptCategory.DUPLICATE: 1,
        AttemptCategory.GENERATION: 0,
        AttemptCategory.LLM_PROPOSED: 0,
    }


def test_budget_categories_match_seeding_result() -> None:
    budget = _big_budget()
    _, result = build_seeded_population(
        ["a:1", "a:2", "bad:1"],
        n=3,
        num_neighbors_cap=8,
        draw_random=_sequence_drawer(["b:1", "bad:2", "c:1"]),
        is_valid=_is_valid,
        flatten_key=_key,
        budget=budget,
        draw_cap=10,
    )
    assert budget.spent() == result.attempted
    assert budget.spent_by(AttemptCategory.INITIAL_POPULATION) == result.valid
    assert budget.spent_by(AttemptCategory.DUPLICATE) == result.duplicates
    assert budget.spent_by(AttemptCategory.INVALID) == result.invalid


def test_equal_budget_stops_at_same_total_regardless_of_success() -> None:
    limit = 20
    unreachable_n = 1000  # forces the global budget to be the binding constraint

    # Arm A: every draw is a fresh valid config -> all 20 attempts succeed.
    budget_a = AttemptBudget(limit)
    with pytest.raises(PopulationUnderfilled) as err_a:
        build_seeded_population(
            [],
            n=unreachable_n,
            num_neighbors_cap=8,
            draw_random=_sequence_drawer([f"a{i}:1" for i in range(limit)]),
            is_valid=_is_valid,
            flatten_key=_key,
            budget=budget_a,
            draw_cap=10_000,
        )

    # Arm B: every draw is invalid -> zero attempts succeed.
    budget_b = AttemptBudget(limit)
    with pytest.raises(PopulationUnderfilled) as err_b:
        build_seeded_population(
            [],
            n=unreachable_n,
            num_neighbors_cap=8,
            draw_random=_sequence_drawer([f"bad:{i}" for i in range(limit)]),
            is_valid=_is_valid,
            flatten_key=_key,
            budget=budget_b,
            draw_cap=10_000,
        )

    # Identical limit -> identical total attempts, even though success differs.
    assert budget_a.spent() == limit
    assert budget_b.spent() == limit
    assert budget_a.spent_by(AttemptCategory.INITIAL_POPULATION) == limit
    assert budget_b.spent_by(AttemptCategory.INITIAL_POPULATION) == 0
    assert err_a.value.budget_exhausted is True
    assert err_b.value.budget_exhausted is True


def test_returns_exactly_n_when_seeds_alone_suffice() -> None:
    budget = _big_budget()
    population, result = build_seeded_population(
        ["a:1", "b:1", "c:1"],
        n=2,
        num_neighbors_cap=8,
        draw_random=_sequence_drawer([]),  # never called: seeds already fill N
        is_valid=_is_valid,
        flatten_key=_key,
        budget=budget,
        draw_cap=10,
    )
    assert population == ["a:1", "b:1"]
    assert result.random_draws == 0
    assert result.valid == 2
