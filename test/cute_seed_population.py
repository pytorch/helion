"""Shared collective seed ordering and initial-population checks."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from test.cute_population_contracts import checked_initial_population

from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch

if TYPE_CHECKING:
    import helion
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel


def _legacy_order(
    spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
) -> list[helion.Config]:
    return seeds


def _legacy_registered_seeds(
    bound: BoundKernel, *, require_no_parameters: bool = False
) -> list[helion.Config]:
    assert bound.host_function is not None
    with (
        bound.env,
        patch.object(
            CuteCollectiveMatmulHeuristic,
            "_interleave_bounded_k_seeds",
            side_effect=_legacy_order,
        ) as interleaved,
    ):
        legacy = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    if require_no_parameters:
        assert all(not call.args[2] for call in interleaved.call_args_list)
    # Other registered heuristics retain their original positions. Replace
    # only this heuristic's permutation when reproducing the old population.
    members = set(legacy)
    selected = [
        seed for seed in bound.config_spec.compiler_seed_configs if seed in members
    ]
    assert len(selected) == len(legacy)
    iterator = iter(legacy)
    return [
        next(iterator) if seed in members else seed
        for seed in bound.config_spec.compiler_seed_configs
    ]


def _initial_population(
    bound: BoundKernel,
    args: tuple[object, ...],
    *,
    seeds: list[helion.Config] | None = None,
    count: int | None = None,
    user_seeds: list[helion.Config] | None = None,
    random_seed: int = 46291,
    assert_raw_canonical: bool = True,
) -> list[helion.Config]:
    """Compare seeded populations without changing the caller's RNG state.

    Raw mutation rows can normalize in some families, so callers opt out of
    requiring those raw flats to be canonical. Effective Configs always retain
    the same uniqueness and flatten/unflatten checks.
    """
    state = random.getstate()
    try:
        random.seed(random_seed)
        profile = get_effort_profile("full").pattern_search
        assert profile is not None
        with (
            bound.env,
            patch.object(
                bound.config_spec,
                "compiler_seed_configs",
                bound.config_spec.compiler_seed_configs if seeds is None else seeds,
            ),
            patch.object(bound.settings, "autotune_seed_configs", user_seeds),
        ):
            search = PatternSearch(
                bound,
                args,
                initial_population=profile.initial_population
                if count is None
                else count,
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            )
            flat = checked_initial_population(search)
            result = [search.config_gen.unflatten(value) for value in flat]
            if assert_raw_canonical:
                assert [search.config_gen.flatten(config) for config in result] == flat
            assert len(result) == len(set(result))
            assert all(
                search.config_gen.unflatten(search.config_gen.flatten(config)) == config
                for config in result
            )
            assert result[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            return result
    finally:
        random.setstate(state)
