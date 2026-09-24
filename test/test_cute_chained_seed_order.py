from __future__ import annotations

from collections import Counter
import itertools
from typing import cast
from unittest.mock import patch

import pytest

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_chained_tcgen05_config import _bound
from test.test_cute_chained_tcgen05_config import _cpu_support  # noqa: F401
from test.test_cute_chained_tcgen05_config import _without_early_release_seed

from helion._compiler.autotuner_heuristics.chained_seed_order import (
    order_chained_seed_configs,
)
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.runtime.config import Config

pytestmark = skipUnlessBackends(["cute"])


def _seed(
    family: str, warps: int, width: int, child: int = 0, pid: str = "flat"
) -> Config:
    return Config.from_dict(
        {
            "block_sizes": [128, width],
            "num_warps": warps,
            "pid_type": pid,
            "cute_chained_mma_schedule": family,
            "cute_chained_pointwise_vectorize": bool(child & 1),
            "cute_chained_auxiliary_cache": bool(child & 2),
        }
    )


def _parent(seed: Config) -> tuple[str, int, tuple[int, ...], str]:
    return (
        cast("str", seed.config["cute_chained_mma_schedule"]),
        seed.num_warps,
        tuple(seed.block_sizes),
        seed.pid_type,
    )


@pytest.mark.parametrize("count", [0, 1, 2, 17])
def test_order_preserves_duplicate_objects_and_input(count: int) -> None:
    seed = _seed("coalesced", 4, 32)
    original = [seed] * count
    result = order_chained_seed_configs(original)
    assert result is not original
    assert result == original
    assert all(item is seed for item in result)
    assert original == [seed] * count


def test_hierarchy_visits_all_parents_before_optional_children() -> None:
    seeds = list(
        itertools.starmap(
            _seed,
            itertools.product(
                ("coalesced", "tcgen05_tmem"), (4, 8), (32, 64), range(4)
            ),
        )
    )
    before = [dict(seed.config) for seed in seeds]
    ordered = order_chained_seed_configs(seeds)
    assert Counter(map(id, ordered)) == Counter(map(id, seeds))
    assert ordered[0] is seeds[0]
    assert [seed.config for seed in seeds] == before
    expected = [
        (family, warps, (128, width), "flat")
        for width, warps, family in itertools.product(
            (32, 64), (4, 8), ("coalesced", "tcgen05_tmem")
        )
    ]
    assert [_parent(seed) for seed in ordered[:8]] == expected
    for parent in expected:
        assert [seed for seed in ordered if _parent(seed) == parent] == [
            seed for seed in seeds if _parent(seed) == parent
        ]
    assert [_parent(seed) for seed in ordered[8:16]] == expected


def test_uneven_families_and_pid_geometry_are_not_dropped() -> None:
    seeds = [_seed("coalesced", 4, 32, child) for child in range(4)] + [
        _seed("coalesced", 4, 32, pid="persistent"),
        _seed("tcgen05_tmem", 4, 64),
        _seed("cp_async", 8, 16),
    ]
    ordered = order_chained_seed_configs(seeds)
    assert ordered[:4] == [seeds[0], seeds[5], seeds[6], seeds[4]]
    assert Counter(ordered) == Counter(seeds)


@pytest.mark.parametrize(
    ("n", "row_block", "scan"),
    [(64, None, False), (256, None, False), (64, 128, False), (64, None, True)],
)
def test_real_specs_keep_seed_multisets_defaults_and_breadth(
    n: int, row_block: int | None, scan: bool
) -> None:
    bound = _bound(n=n, row_block=row_block, scan=scan)
    with bound.env:
        assert bound.host_function is not None
        device_ir = bound.host_function.device_ir
        with patch(
            "helion._compiler.autotuner_heuristics.cute.order_chained_seed_configs",
            side_effect=lambda seeds: seeds,
        ):
            old = CuteChainedMatmulHeuristic.get_seed_configs(bound.env, device_ir)
        new = CuteChainedMatmulHeuristic.get_seed_configs(bound.env, device_ir)
        assert old is not None and new is not None
        old = _without_early_release_seed(old)
        new = _without_early_release_seed(new)
        assert Counter(old) == Counter(new)
        assert new == order_chained_seed_configs(old)
        assert new[0] == old[0]
        assert (
            CuteChainedMatmulHeuristic.get_seed_config(bound.env, device_ir) == old[0]
        )
        spec = bound.config_spec
        actual = spec.compiler_seed_configs
        try:
            spec.compiler_seed_configs = old
            old_generation = ConfigGeneration(spec)
            default = old_generation.default_flat()
            old_set = {config for _, config in old_generation.seed_flat_config_pairs()}
            spec.compiler_seed_configs = new
            generation = ConfigGeneration(spec)
            assert generation.default_flat() == default
            assert {
                config for _, config in generation.seed_flat_config_pairs()
            } == old_set
            population_size = min(100, len(old_set))
            with (
                patch.object(
                    generation, "random_flat", side_effect=AssertionError("random fill")
                ),
                patch.object(
                    generation,
                    "biased_random_flat",
                    side_effect=AssertionError("random fill"),
                ),
            ):
                population = [
                    generation.unflatten(flat)
                    for flat in generation.random_population_flat(population_size)
                ]
            assert len(population) == population_size
            expected_parents = {_parent(config) for config in old_set}
            actual_parents = {_parent(config) for config in population}
            # Wider schemas can have more structural parents than the unchanged
            # 100-candidate population. Cover as many distinct parents as fit,
            # without claiming that every cross-product fits in that budget.
            assert actual_parents <= expected_parents
            assert len(actual_parents) == min(population_size, len(expected_parents))
            assert {parent[:2] for parent in actual_parents} == {
                parent[:2] for parent in expected_parents
            }
            assert {
                parent for parent in actual_parents if parent[0] == "tcgen05_tmem"
            } == {parent for parent in expected_parents if parent[0] == "tcgen05_tmem"}
            wanted = Config(
                block_sizes=[64] if row_block else [128, 64],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
            )
            _, canonical = generation.canonicalize_flat(generation.flatten(wanted))
            assert canonical in population
        finally:
            spec.compiler_seed_configs = actual


@pytest.mark.parametrize("size", [1, 7, 13, 24, 60, 100])
def test_population_budget_and_explicit_user_priority_unchanged(size: int) -> None:
    bound = _bound(n=64)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        user = Config(block_sizes=[32, 64], num_warps=2)
        _, normalized_user = generation.canonicalize_flat(generation.flatten(user))
        with (
            patch.object(
                generation, "random_flat", side_effect=AssertionError("random fill")
            ),
            patch.object(
                generation,
                "biased_random_flat",
                side_effect=AssertionError("random fill"),
            ),
        ):
            population = generation.random_population_flat(
                size, user_seed_configs=[user]
            )
        assert len(population) == size
        assert population[0] == generation.default_flat()
        if size > 1:
            assert generation.unflatten(population[1]) == normalized_user
