from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from typing import TypeVar
from typing import cast

import pytest

from helion import exc
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.candidate_budget import AttemptBudget
from helion.autotuner.candidate_budget import AttemptCategory
from helion.autotuner.candidate_budget import CandidatePopulationUnderfilled
from helion.autotuner.candidate_budget import random_replacement_draw_cap
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.rag.seeding import AttemptBudget as SeedingAttemptBudget
from helion.autotuner.rag.seeding import AttemptCategory as SeedingAttemptCategory
from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.runtime.settings import Settings


class _ConfigGeneration:
    def flatten(self, config: str) -> list[str]:
        return [config]

    def unflatten(self, flat: list[str]) -> str:
        config = flat[0]
        if config == "invalid":
            raise exc.InvalidConfig("invalid test config")
        return config

    def random_population_flat(self, n: int) -> list[list[str]]:
        del n
        return [["default"], ["random"], ["random"], ["invalid"]]


_SearchT = TypeVar("_SearchT", bound=PopulationBasedSearch)


def _search(budget: AttemptBudget, cls: type[_SearchT]) -> _SearchT:
    search = cls.__new__(cls)
    search.config_gen = cast("Any", _ConfigGeneration())
    search._candidate_attempt_budget = budget
    search._candidate_attempt_configs = set()
    search._candidate_initial_population_open = True
    search._precounted_initial_population_configs = set()
    search._candidate_attempt_categories = {}
    search._candidate_sources = {}
    search._fixed_initial_population_flat = None
    return search


def test_attempt_budget_stops_at_limit_without_overshoot() -> None:
    budget = AttemptBudget(3)

    assert budget.record(AttemptCategory.INITIAL_POPULATION, 2)
    assert budget.record(AttemptCategory.INVALID)
    assert not budget.record(AttemptCategory.GENERATION)

    assert budget.spent() == 3
    assert budget.remaining() == 0
    assert budget.exhausted
    assert budget.by_category() == {
        AttemptCategory.INITIAL_POPULATION: 2,
        AttemptCategory.INVALID: 1,
        AttemptCategory.DUPLICATE: 0,
        AttemptCategory.GENERATION: 0,
        AttemptCategory.LLM_PROPOSED: 0,
    }


def test_seed_builder_reexports_the_shared_budget_types() -> None:
    assert SeedingAttemptBudget is AttemptBudget
    assert SeedingAttemptCategory is AttemptCategory


def test_attempt_budget_can_be_disabled_without_losing_counts() -> None:
    budget = AttemptBudget(None)

    assert budget.record(AttemptCategory.GENERATION, 10)

    assert budget.limit is None
    assert budget.remaining() is None
    assert not budget.exhausted
    assert budget.spent() == 10


@pytest.mark.parametrize("limit", [0, -1])
def test_attempt_budget_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        AttemptBudget(limit)


@pytest.mark.parametrize("n", [0, -1])
def test_attempt_budget_rejects_non_positive_record_count(n: int) -> None:
    budget = AttemptBudget(3)

    with pytest.raises(ValueError, match="positive"):
        budget.record(AttemptCategory.DUPLICATE, n)


def test_candidate_attempt_limit_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT", raising=False)

    assert Settings().autotune_candidate_attempt_limit is None


def test_candidate_attempt_limit_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT", "17")

    assert Settings().autotune_candidate_attempt_limit == 17


def test_controlled_comparison_can_disable_trajectory_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_DISABLE_TRAJECTORY_EARLY_STOP", "1")

    assert Settings().autotune_disable_trajectory_early_stop is True


def test_confirmation_correctness_tolerances_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HELION_AUTOTUNE_BASELINE_RTOL", "1e-5")
    monkeypatch.setenv("HELION_AUTOTUNE_BASELINE_ATOL", "1e-6")

    settings = Settings()

    assert settings.autotune_baseline_rtol == 1e-5
    assert settings.autotune_baseline_atol == 1e-6


@pytest.mark.parametrize("limit", [0, -1])
def test_settings_reject_non_positive_candidate_attempt_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="autotune_candidate_attempt_limit.*positive"):
        Settings(autotune_candidate_attempt_limit=limit)


def test_search_shares_cap_across_initial_rejections_and_generation() -> None:
    budget = AttemptBudget(4)
    search = _search(budget, PopulationBasedSearch)

    assert search.make_unbenchmarked(["first"]) is not None
    assert search.make_unbenchmarked(["invalid"]) is None
    assert search.make_unbenchmarked(["first"]) is None
    search._finish_initial_candidate_attempts()
    assert search.make_unbenchmarked(["second"]) is not None
    assert search.make_unbenchmarked(["third"]) is None

    assert budget.spent() == 4
    assert budget.by_category() == {
        AttemptCategory.INITIAL_POPULATION: 1,
        AttemptCategory.INVALID: 1,
        AttemptCategory.DUPLICATE: 1,
        AttemptCategory.GENERATION: 1,
        AttemptCategory.LLM_PROPOSED: 0,
    }


def test_search_default_off_keeps_duplicate_admission_behavior() -> None:
    budget = AttemptBudget(None)
    search = _search(budget, PopulationBasedSearch)

    assert search.make_unbenchmarked(["same"]) is not None
    assert search.make_unbenchmarked(["same"]) is not None

    assert budget.spent() == 0


def test_fixed_population_can_reuse_already_charged_budget() -> None:
    budget = AttemptBudget(2)
    assert budget.record(AttemptCategory.INITIAL_POPULATION, 2)
    search = _search(AttemptBudget(None), PopulationBasedSearch)
    search.set_candidate_attempt_budget(budget)
    search.set_fixed_initial_population_configs(
        cast("Any", ["first", "second"]), attempts_already_recorded=True
    )

    members = [
        search.make_unbenchmarked(flat)
        for flat in search.fixed_initial_population_flat() or []
    ]

    assert all(member is not None for member in members)
    assert budget.spent() == 2


def test_budget_cannot_be_reset_after_search_attempts_start() -> None:
    search = _search(AttemptBudget(2), PopulationBasedSearch)
    assert search.make_unbenchmarked(["first"]) is not None

    with pytest.raises(RuntimeError, match="already started"):
        search.set_candidate_attempt_budget(AttemptBudget(10))


def test_lfbo_generation_uses_base_search_budget() -> None:
    budget = AttemptBudget(2)
    search = _search(budget, LFBOTreeSearch)
    search._finish_initial_candidate_attempts()

    assert search.make_unbenchmarked(["first"]) is not None
    assert search.make_unbenchmarked(["first"]) is None
    assert search.make_unbenchmarked(["second"]) is None

    assert budget.spent_by(AttemptCategory.GENERATION) == 1
    assert budget.spent_by(AttemptCategory.DUPLICATE) == 1


def test_llm_proposals_share_remaining_budget_and_count_duplicates() -> None:
    budget = AttemptBudget(3)
    assert budget.record(AttemptCategory.INITIAL_POPULATION)
    search = _search(budget, LLMGuidedSearch)
    cast("Any", search._candidate_attempt_configs).add("seed")
    seen = {repr("seed")}

    result = search._dedupe_new_configs(cast("Any", ["seed", "new", "over-cap"]), seen)

    assert result == ["new"]
    assert seen == {repr("seed"), repr("new")}
    assert budget.spent() == 3
    assert budget.spent_by(AttemptCategory.DUPLICATE) == 1
    assert budget.spent_by(AttemptCategory.LLM_PROPOSED) == 1


def test_llm_initial_seeds_count_rejected_draws_against_same_cap() -> None:
    budget = AttemptBudget(3)
    search = _search(budget, LLMGuidedSearch)
    search.config_spec = cast("Any", SimpleNamespace(default_config=lambda: "default"))
    search.initial_random_configs = 3

    with pytest.raises(CandidatePopulationUnderfilled, match="2/4"):
        search._build_seed_configs()

    assert budget.spent() == 3
    assert budget.spent_by(AttemptCategory.INITIAL_POPULATION) == 2
    assert budget.spent_by(AttemptCategory.DUPLICATE) == 1


def test_llm_uses_precounted_fixed_initial_population() -> None:
    budget = AttemptBudget(2)
    assert budget.record(AttemptCategory.INITIAL_POPULATION, 2)
    search = _search(budget, LLMGuidedSearch)
    search.config_spec = cast("Any", SimpleNamespace(default_config=lambda: "default"))
    search.initial_random_configs = 3
    search.set_fixed_initial_population_configs(
        cast("Any", ["fixed-a", "fixed-b"]), attempts_already_recorded=True
    )

    assert search._build_seed_configs() == ["fixed-a", "fixed-b"]
    assert budget.spent() == 2


def test_budgeted_range_stops_when_candidate_cap_is_exhausted() -> None:
    budget = AttemptBudget(1)
    assert budget.record(AttemptCategory.INITIAL_POPULATION)
    search = _search(budget, PopulationBasedSearch)
    search.settings = cast("Any", SimpleNamespace(autotune_budget_seconds=None))
    search._autotune_budget_start = None

    assert list(search._budgeted_range(5)) == []


def test_eager_generation_is_limited_to_remaining_attempts() -> None:
    budget = AttemptBudget(5)
    assert budget.record(AttemptCategory.INITIAL_POPULATION, 3)
    search = _search(budget, PopulationBasedSearch)

    assert search._candidate_generation_limit(10) == 2


def test_unchanged_lfbo_draws_are_charged_as_duplicates() -> None:
    budget = AttemptBudget(3)
    search = _search(budget, LFBOPatternSearch)
    search.num_neighbors = 20
    search.num_neighbors_cap = -1
    search.radius = 1
    search.config_gen = cast(
        "Any",
        SimpleNamespace(
            overridden_flat_indices=set(),
            block_size_indices=[],
            num_warps_index=-1,
            flat_spec=[],
        ),
    )

    assert search._generate_neighbors([]) == []
    assert budget.spent() == 3
    assert budget.spent_by(AttemptCategory.DUPLICATE) == 3


def test_capped_initial_population_refills_rejected_draws_to_exact_size() -> None:
    class RefillConfigGeneration(_ConfigGeneration):
        def random_flat(self) -> list[str]:
            return ["replacement"]

    budget = AttemptBudget(4)
    search = _search(budget, PopulationBasedSearch)
    search.config_gen = cast("Any", RefillConfigGeneration())

    population, _ = search.make_initial_population(
        [["default"], ["default"], ["invalid"]], target_size=2
    )

    assert [member.config for member in population] == ["default", "replacement"]
    assert budget.spent() == 4


def test_capped_initial_population_fails_instead_of_silently_shrinking() -> None:
    budget = AttemptBudget(3)
    search = _search(budget, PopulationBasedSearch)

    with pytest.raises(CandidatePopulationUnderfilled, match="1/2"):
        search.make_initial_population(
            [["default"], ["default"], ["invalid"]], target_size=2
        )


def test_initial_replacement_draws_stop_at_shared_frozen_cap() -> None:
    class InvalidReplacementGeneration(_ConfigGeneration):
        def random_flat(self) -> list[str]:
            return ["invalid"]

    budget = AttemptBudget(1_000)
    search = _search(budget, PopulationBasedSearch)
    search.config_gen = cast("Any", InvalidReplacementGeneration())

    with pytest.raises(CandidatePopulationUnderfilled, match="1/2"):
        search.make_initial_population([["default"]], target_size=2)

    assert budget.spent() == 1 + random_replacement_draw_cap(2)


def test_shared_attempt_state_spans_stages_with_cap_and_cross_stage_dedup() -> None:
    """Two searches adopting one state share the ceiling and dedup across stages."""
    from helion.autotuner.candidate_budget import SharedAttemptState

    shared = SharedAttemptState(AttemptBudget(4))
    stage_one = _search(AttemptBudget(None), PopulationBasedSearch)
    stage_two = _search(AttemptBudget(None), PopulationBasedSearch)
    stage_one.adopt_shared_attempt_state(shared)
    stage_two.adopt_shared_attempt_state(shared)

    # Stage one attempts two unique configs against the shared budget.
    assert stage_one.make_unbenchmarked(["a"]) is not None
    assert stage_one.make_unbenchmarked(["b"]) is not None
    # Stage two sees stage one's config as a duplicate (shared dedup set) ...
    assert stage_two.make_unbenchmarked(["a"]) is None
    # ... and shares the same 4-attempt ceiling: one fresh config exhausts it.
    assert stage_two.make_unbenchmarked(["c"]) is not None
    assert stage_two.make_unbenchmarked(["d"]) is None

    assert shared.budget.spent() == 4
    assert shared.budget.spent_by(AttemptCategory.DUPLICATE) == 1
    assert stage_one._candidate_attempt_configs is stage_two._candidate_attempt_configs
    assert stage_one.candidate_attempt_budget is shared.budget


def test_adopt_shared_attempt_state_rejects_after_attempts_started() -> None:
    from helion.autotuner.candidate_budget import SharedAttemptState

    search = _search(AttemptBudget(3), PopulationBasedSearch)
    assert search.make_unbenchmarked(["first"]) is not None

    with pytest.raises(RuntimeError, match="already started"):
        search.adopt_shared_attempt_state(SharedAttemptState(AttemptBudget(5)))


def test_adopt_shared_attempt_state_shares_one_collector() -> None:
    from helion.autotuner.candidate_budget import SharedAttemptState
    from helion.autotuner.rag.instrumentation import InstrumentationCollector

    collector = InstrumentationCollector()
    shared = SharedAttemptState(AttemptBudget(None), collector=collector)
    stage_one = _search(AttemptBudget(None), PopulationBasedSearch)
    stage_two = _search(AttemptBudget(None), PopulationBasedSearch)
    stage_one.adopt_shared_attempt_state(shared)
    stage_two.adopt_shared_attempt_state(shared)

    assert stage_one._attempt_instrumentation is collector
    assert stage_two._attempt_instrumentation is collector
