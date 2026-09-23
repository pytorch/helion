from __future__ import annotations

import ast
from copy import deepcopy
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.rms_norm import rms_norm_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_host_paired_sum_seeds import _population
from test.test_cute_host_single_sum import _arguments
from test.test_cute_host_single_sum import _single

import helion
from helion._compiler.cute.host_paired_sum import PAIRED_SUM_KEY
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

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


def _original(static: bool) -> tuple[BoundKernel, tuple[torch.Tensor, ...]]:
    kernel = helion.kernel(
        rms_norm_bwd.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
    )
    arguments = (
        torch.empty((256, 1024), dtype=torch.float16),
        torch.empty((256, 1024), dtype=torch.float16, requires_grad=True),
        torch.empty(1024, dtype=torch.float16, requires_grad=True),
        torch.empty((256, 1), dtype=torch.float32),
    )
    return _cpu_bind(kernel, arguments), arguments


@pytest.mark.parametrize("static", (True, False))
def test_original_single_return_actual_initial100_reaches_both_layouts(
    static: bool,
) -> None:
    bound, arguments = _original(static)
    population = _population(bound, arguments)
    assert population == _population(bound, arguments)
    assert population[0][PAIRED_SUM_KEY] == "off"
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
    for layout in ("mapped", "narrow"):
        index = next(
            i for i, config in enumerate(population) if config[PAIRED_SUM_KEY] == layout
        )
        assert index < 5
        selected = population[index]
        with bound.env:
            flat, normalized = generation.canonicalize_flat(
                generation.flatten(selected)
            )
            assert generation.unflatten(flat) == normalized == selected
        code = bound.to_code(selected)
        assert "_cute_try_single_sum_cast(" in code
        off = deepcopy(selected)
        off.config[PAIRED_SUM_KEY] = "off"
        before, after = ast.parse(bound.to_code(off)), ast.parse(code)
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
        for seed in bound.config_spec.compiler_seed_configs:
            _, normalized = generation.canonicalize_flat(generation.flatten(seed))
            assert normalized in population


def test_generic_single_seeds_have_independent_nested_configs() -> None:
    arguments = _arguments()
    bound = _cpu_bind(_single, arguments)
    assert {config[PAIRED_SUM_KEY] for config in _population(bound, arguments)[:5]} == {
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
                previous = value[0]
                value[0] = "independent mutation"
                assert all(
                    sibling.config == before[j]
                    for j, sibling in enumerate(seeds)
                    if j != index
                )
                value[0] = previous
    assert [seed.config for seed in seeds] == before


def test_disabling_heuristics_preserves_explicit_single_choice() -> None:
    kernel = helion.kernel(
        _single.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    bound = _cpu_bind(kernel, _arguments())
    assert bound.config_spec.cute_host_paired_sum_available
    assert not bound.config_spec.compiler_seed_configs
    with bound.env:
        config = bound.config_spec.default_config()
    config.config[PAIRED_SUM_KEY] = "narrow"
    assert "_cute_try_single_sum_cast(" in bound.to_code(config)
