from __future__ import annotations

import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import checked_initial_population
from test.test_cute_collective_tmem_operand_codegen import _mamba_kernel
from test.test_cute_collective_tmem_operand_codegen import _sizes
from test.test_cute_collective_tmem_seed import _dependent_contractions
from test.test_cute_collective_tmem_seed import _dependent_inputs

from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    import helion
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda_init() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _legacy_order(
    spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
) -> list[helion.Config]:
    # None of these contractions has a config-owned K chunk. The reviewed
    # bounded-K interleaver previously returned this list unchanged.
    assert not parameters
    return seeds


def _legacy_registered_seeds(bound: BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with (
        bound.env,
        patch.object(
            CuteCollectiveMatmulHeuristic,
            "_interleave_bounded_k_seeds",
            side_effect=_legacy_order,
        ),
    ):
        legacy = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
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
    args: tuple[torch.Tensor, ...],
    *,
    seeds: list[helion.Config] | None = None,
    count: int | None = None,
    user_seeds: list[helion.Config] | None = None,
) -> list[helion.Config]:
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    random_state = random.getstate()
    try:
        random.seed(46291)
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
                initial_population=(
                    profile.initial_population if count is None else count
                ),
                initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
            )
            flat = checked_initial_population(search)
            result = [search.config_gen.unflatten(value) for value in flat]
            assert result[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            return result
    finally:
        random.setstate(random_state)


def _complete_schedule(config: helion.Config) -> bool:
    return (
        all(
            config.get(key, False)
            for key in (
                "cute_collective_tmem_a",
                "cute_collective_native_seeded",
                "cute_collective_tmem_seed",
                "cute_proven_bounds",
            )
        )
        and config.get("cute_collective_compute") == "tcgen05"
        and config.get("cute_collective_recipe") == "vector_unrolled"
        and config.get("cute_collective_epilogue") == "vector_unrolled"
    )


@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_full_initial_population_reaches_complete_tmem_operand_schedule(
    shape: int, dtype: torch.dtype, static_shapes: bool
) -> None:
    args = tuple(torch.empty(size, dtype=dtype) for size in _sizes(shape))
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), args)
    old_seeds = _legacy_registered_seeds(bound)
    new_seeds = bound.config_spec.compiler_seed_configs
    assert old_seeds[0] == new_seeds[0]
    assert len(old_seeds) == len(new_seeds)
    assert set(old_seeds) == set(new_seeds)
    before = _initial_population(bound, args, seeds=old_seeds)
    after = _initial_population(bound, args)
    assert len(before) == len(after)
    assert before[0] == after[0]
    assert after == _initial_population(bound, args)
    if shape == 2:
        assert not any(_complete_schedule(config) for config in before)
    complete = [config for config in after if _complete_schedule(config)]
    assert complete
    if shape:
        assert any(config.block_sizes == [64, 64, 64] for config in complete)
    source = bound.to_code(complete[0])
    assert source.count("OperandSource.TMEM") == 1
    assert source.count("OperandSource.SMEM") == 1
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert "collective_tmem_seed_values" in source
    assert "mul.rn.f32" in source


@pytest.fixture(scope="module")
def dependent_case() -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    args = _dependent_inputs(torch.bfloat16, "cpu")
    return _cpu_bind(_dependent_contractions, args), args


@pytest.mark.parametrize("count", [2, 7, 30, 100])
def test_tmem_schedule_order_preserves_budget_and_explicit_user_priority(
    dependent_case: tuple[BoundKernel, tuple[torch.Tensor, ...]], count: int
) -> None:
    bound, args = dependent_case
    explicit = next(
        config
        for config in reversed(bound.config_spec.compiler_seed_configs)
        if config.get("cute_collective_tmem_a")
    )
    population = _initial_population(bound, args, count=count, user_seeds=[explicit])
    # The helper checks exactly count legacy rows plus declared coverage only.
    with bound.env:
        generation = bound.config_spec.create_config_generation()
        _flat, normalized = generation.canonicalize_flat(generation.flatten(explicit))
    assert population[1] == normalized


def test_unrelated_seed_configs_retain_their_exact_order(
    dependent_case: tuple[BoundKernel, tuple[torch.Tensor, ...]],
) -> None:
    bound, _args = dependent_case
    seeds = [
        config
        for config in _legacy_registered_seeds(bound)
        if not config.get("cute_collective_tmem_a", False)
    ]
    assert (
        CuteCollectiveMatmulHeuristic._interleave_bounded_k_seeds(
            bound.config_spec, seeds, ()
        )
        is seeds
    )
