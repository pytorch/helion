from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _initial_population as _shared_population
from test.cute_population_contracts import _legacy_registered_seeds
from test.test_cute_collective_tmem_operand_codegen import _mamba_kernel
from test.test_cute_collective_tmem_operand_codegen import _sizes
from test.test_cute_collective_tmem_seed import _dependent_contractions
from test.test_cute_collective_tmem_seed import _dependent_inputs

from helion._testing import skipUnlessBackends

_initial_population = partial(_shared_population, random_seed=46291)

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    import helion
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda_init() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


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
