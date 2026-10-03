from __future__ import annotations

from copy import deepcopy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import checked_initial_population
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


def _population(
    bound: BoundKernel, arguments: tuple[torch.Tensor, ...]
) -> list[helion.Config]:
    state = random.getstate()
    try:
        with bound.env:
            search = LFBOTreeSearch(
                bound,
                arguments,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            assert search.initial_population == 100
            assert (
                search.initial_population_strategy
                is InitialPopulationStrategy.FROM_RANDOM
            )
            random.seed(2026091539)
            result = [
                search.config_gen.unflatten(flat)
                for flat in checked_initial_population(search)
            ]
            assert result[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            assert len(result) == len(set(result))
            return result
    finally:
        random.setstate(state)


@pytest.mark.parametrize(
    "batch,rows,k,n", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
def test_real_full_lfbo_population_reaches_packets_and_retains_other_families(
    batch: int, rows: int, k: int, n: int
) -> None:
    arguments = _jagged_inputs(batch=batch, rows=rows, k=k, n=n)
    bound = _cpu_bind(_kernel(), arguments)
    population = _population(bound, arguments)
    assert population == _population(bound, arguments)
    packet_seeds = [
        config
        for config in population
        if config.get("cute_collective_operand_packets", False)
    ]
    assert packet_seeds
    assert any(
        "operand_packet_destination" in bound.to_code(config) for config in packet_seeds
    )
    for compute in ("warp", "tcgen05"):
        assert any(
            config.get("cute_collective_mma")
            and config.get("cute_collective_compute") == compute
            and config.get("cute_collective_recipe") == "scalar"
            for config in population
        )
    for stages in (2, 4):
        assert any(
            config.get("cute_collective_mma")
            and config.get("cute_collective_compute") == "warp"
            and config.get("cute_collective_copy") == "async_cached"
            and config.get("cute_collective_stages") == stages
            for config in population
        )
    assert any(not config.get("cute_collective_mma", False) for config in population)
    assert any(
        config.get("cute_collective_recipe") == "vector_unrolled"
        and not config.get("cute_collective_operand_packets", False)
        for config in population
    )


def test_ordinary_computed_batched_matmul_reaches_packets() -> None:
    arguments = (torch.empty((3, 130, 64)), torch.empty((3, 64, 70)))
    bound = _cpu_bind(_kernel(batched=True), arguments)
    choices = [
        config
        for config in _population(bound, arguments)
        if config.get("cute_collective_operand_packets", False)
    ]
    assert choices and any(
        "operand_packet_destination" in bound.to_code(config) for config in choices
    )


@pytest.mark.parametrize(
    "batch,rows,k,n", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
def test_direct_packet_seed_mutation_cannot_change_scalar_or_packet_siblings(
    batch: int, rows: int, k: int, n: int
) -> None:
    arguments = _jagged_inputs(batch=batch, rows=rows, k=k, n=n)
    bound = _cpu_bind(_kernel(), arguments)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    original = deepcopy([seed.config for seed in seeds])
    packet_indices = [
        index
        for index, seed in enumerate(seeds)
        if seed.get("cute_collective_operand_packets", False)
    ]
    assert packet_indices
    for index in packet_indices:
        packet = seeds[index]
        control = {
            key: value
            for key, value in packet.config.items()
            if key != "cute_collective_operand_packets"
        }
        assert any(seed.config == control for seed in seeds)
        for field in (
            "block_sizes",
            "num_threads",
            "cute_vector_widths",
            "cute_lane_layouts",
        ):
            values = packet.config[field]
            assert isinstance(values, list) and values
            first = values[0]
            values[0] = "packet mutation witness"
            assert all(
                sibling.config == original[sibling_index]
                for sibling_index, sibling in enumerate(seeds)
                if sibling_index != index
            ), (index, field)
            values[0] = first
    assert [seed.config for seed in seeds] == original


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise(a: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for rows, columns in hl.tile(a.shape):
        out[rows, columns] = a[rows, columns] + 1
    return out


@pytest.mark.parametrize("kind", ["half", "pointwise"])
def test_unrelated_full_populations_do_not_get_packet_seeds(kind: str) -> None:
    if kind == "half":
        arguments = (
            torch.empty((3, 130, 64), dtype=torch.bfloat16),
            torch.empty((3, 64, 70), dtype=torch.bfloat16),
        )
        kernel = _kernel(batched=True)
    else:
        arguments = (torch.empty((130, 70)),)
        kernel = _pointwise
    bound = _cpu_bind(kernel, arguments)
    assert all(
        not config.get("cute_collective_operand_packets", False)
        for config in bound.config_spec.compiler_seed_configs
    )
    assert all(
        not config.get("cute_collective_operand_packets", False)
        for config in _population(bound, arguments)
    )
