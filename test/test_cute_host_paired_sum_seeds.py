from __future__ import annotations

import ast
from copy import deepcopy
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import checked_initial_population
from test.test_cute_host_paired_sum import _args
from test.test_cute_host_paired_sum import _config
from test.test_cute_host_paired_sum import _paired

import helion
from helion._compiler.autotuner_heuristics.cute_host_paired_sum import (
    add_host_sum_seeds,
)
from helion._compiler.cute.host_paired_sum import PAIRED_SUM_KEY
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
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
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _original(static: bool) -> tuple[BoundKernel, tuple[object, ...]]:
    kernel = helion.kernel(
        layer_norm_bwd.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
    )
    arguments = (
        torch.empty((256, 1024), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16, requires_grad=True),
        torch.empty(256, dtype=torch.float32),
        torch.empty(256, dtype=torch.float32),
        torch.empty(1024, dtype=torch.bfloat16, requires_grad=True),
        True,
    )
    return _cpu_bind(kernel, arguments), arguments


def _population(
    bound: BoundKernel, arguments: tuple[object, ...]
) -> list[helion.Config]:
    previous = random.getstate()
    try:
        random.seed(2026091583)
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
            flats = checked_initial_population(search)
            configs = [search.config_gen.unflatten(flat) for flat in flats]
            assert len(configs) == len(set(configs))
            assert [
                search.config_gen.unflatten(search.config_gen.flatten(config))
                for config in configs
            ] == configs
            assert configs[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
            return configs
    finally:
        random.setstate(previous)


@pytest.mark.parametrize("static", (True, False))
def test_original_backward_actual_full_population_reaches_both_host_choices(
    static: bool,
) -> None:
    bound, arguments = _original(static)
    population = _population(bound, arguments)
    assert population == _population(bound, arguments)
    assert population[0][PAIRED_SUM_KEY] == "off"
    indices = {}
    for layout in ("off", "mapped", "narrow"):
        indices[layout] = next(
            i for i, config in enumerate(population) if config[PAIRED_SUM_KEY] == layout
        )
    assert indices["mapped"] < 5 and indices["narrow"] < 5
    for layout in ("mapped", "narrow"):
        code = bound.to_code(population[indices[layout]])
        assert "_cute_try_paired_sum_cast" in code
        control = deepcopy(population[indices[layout]])
        control.config[PAIRED_SUM_KEY] = "off"
        before = ast.parse(bound.to_code(control))
        after = ast.parse(code)
        assert [
            ast.dump(node)
            for node in before.body
            if isinstance(node, ast.FunctionDef) and node.decorator_list
        ] == [
            ast.dump(node)
            for node in after.body
            if isinstance(node, ast.FunctionDef) and node.decorator_list
        ]
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        for seed in bound.config_spec.compiler_seed_configs:
            _, normalized = generator.canonicalize_flat(generator.flatten(seed))
            assert normalized in population


def test_ordinary_pair_gets_valid_reachable_seeds_with_independent_nested_lists() -> (
    None
):
    arguments = _args()
    bound = _cpu_bind(_paired, arguments)
    population = _population(bound, arguments)
    assert {config[PAIRED_SUM_KEY] for config in population[:5]} == {
        "off",
        "mapped",
        "narrow",
    }
    seeds = bound.config_spec.compiler_seed_configs
    before = deepcopy([seed.config for seed in seeds])
    for index, seed in enumerate(seeds):
        if seed.get(PAIRED_SUM_KEY) not in ("mapped", "narrow"):
            continue
        for value in seed.config.values():
            if isinstance(value, list) and value:
                first = value[0]
                value[0] = "mutation witness"
                assert all(
                    sibling.config == before[j]
                    for j, sibling in enumerate(seeds)
                    if j != index
                )
                value[0] = first
    assert [seed.config for seed in seeds] == before


def test_seed_overlay_retains_every_existing_producer_in_order() -> None:
    bound = _cpu_bind(_paired, _args())
    producers = [
        helion.Config(
            block_sizes=[4, 32],
            num_threads=[1, threads],
            cute_collective_compute=family,
        )
        for family in ("warp", "tcgen05", "simt")
        for threads in (32, 64, 128)
    ]
    with bound.env:
        result = add_host_sum_seeds(bound.env, producers)
    assert [seed for seed in result if PAIRED_SUM_KEY not in seed.config] == producers
    assert result[0] is producers[0]
    assert len(result) == len(producers) + 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    for m, n in hl.tile(out.shape):
        acc = hl.full((m, n), 0, torch.float32)
        for k in hl.tile(a.shape[1]):
            acc = hl.dot(a[m, k], b[k, n], acc)
        out[m, n] = acc.to(out.dtype)
    return out


def test_no_pair_does_not_add_a_host_choice_to_matmul_population() -> None:
    arguments = (
        torch.empty((128, 128), dtype=torch.float16),
        torch.empty((128, 128), dtype=torch.float16),
    )
    bound = _cpu_bind(_ordinary_matmul, arguments)
    assert not bound.config_spec.cute_host_paired_sum_available
    assert all(
        PAIRED_SUM_KEY not in config.config for config in _population(bound, arguments)
    )


def test_config_roundtrip_and_search_identity() -> None:
    bound = _cpu_bind(_paired, _args())
    choices = ("off", "mapped", "narrow")
    configs = [_config(bound, layout) for layout in choices]
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        normalized = []
        for layout, config in zip(choices, configs, strict=True):
            flat, restored = generator.canonicalize_flat(generator.flatten(config))
            assert generator.unflatten(flat) == restored
            assert restored[PAIRED_SUM_KEY] == layout
            normalized.append(restored)
        assert len(set(normalized)) == 3
        fields = bound.config_spec._flat_fields()
        assert tuple(fields[PAIRED_SUM_KEY].search_values(100)) == choices
