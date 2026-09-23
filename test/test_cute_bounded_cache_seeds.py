from __future__ import annotations

import ast
import random
from unittest.mock import patch

import pytest
import torch

from .test_cute_bounded_cache_codegen import _bound
from .test_cute_bounded_cache_codegen import _cpu_target
from .test_cute_bounded_cache_codegen import _functions
import helion
from helion._compiler.autotuner_heuristics import HEURISTICS_BY_BACKEND
from helion._compiler.autotuner_heuristics.cute_bounded_loop_cache import (
    CuteBoundedLoopCacheHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _seeds(bound):
    with bound.env:
        seeds = CuteBoundedLoopCacheHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    return seeds


def _population(bound, count=None, explicit=None):
    bound.settings.autotune_random_seed = 2026091542
    with bound.env:
        kwargs = LFBOTreeSearch.get_kwargs_from_profile(
            get_effort_profile("full"), bound.settings
        )
        if count is not None:
            kwargs["initial_population"] = count
        with patch.object(
            bound.settings, "autotune_seed_configs", [explicit] if explicit else None
        ):
            search = LFBOTreeSearch(
                bound, tuple(bound.host_function.params.arguments.values()), **kwargs
            )
            assert (
                search.initial_population_strategy
                == InitialPopulationStrategy.FROM_RANDOM
            )
            random.seed(bound.settings.autotune_random_seed)
            flat = search._generate_initial_population_flat()
            configs = [search.config_gen.unflatten(item) for item in flat]
            assert len(configs) == len(set(configs)) == (count or 100)
            assert [search.config_gen.flatten(config) for config in configs] == flat
            assert configs[0] == search.config_gen.unflatten(
                search.config_gen.default_flat()
            )
    return configs


@pytest.mark.parametrize("width", [256, 1024, 4096])
def test_original_full_random_population_reaches_complete_cache_schedule(width):
    with _cpu_target():
        bound = _bound(shape=(width, width), effort="full")
        seeds = _seeds(bound)
        population = _population(bound)
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            normalized = [
                generation.canonicalize_flat(generation.flatten(seed))[1]
                for seed in seeds
            ]
        assert all(seed in population for seed in normalized)
        primary = normalized[0]
        source = bound.to_code(primary)
    fast = ast.unparse(_functions(source)["_helion_row_softmax_bounded"])
    assert "cute.arch.load(" in fast
    assert "n = cutlass.Int64(cute.size(x, mode=[1]))" in fast
    collectives = [
        node
        for node in ast.walk(ast.parse(fast))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_grouped_reduce_")
    ]
    assert len(collectives) == 2
    if width == 4096:
        assert primary.block_sizes == [1, 4096]
        assert primary.num_threads == [0, 128]
        assert primary["cute_vector_widths"] == [1, 8]
        assert primary["cute_lane_layouts"] == ["blocked", "strided"]


@pytest.mark.parametrize(
    "dtype,vector", [(torch.float16, 8), (torch.bfloat16, 8), (torch.float32, 4)]
)
@pytest.mark.parametrize("static", [False, True])
def test_seed_packet_uses_raw_load_dtype_before_fp32_arithmetic(dtype, vector, static):
    with _cpu_target():
        bound = _bound(dtype=dtype, static=static, shape=(17, 4096))
        primary = _seeds(bound)[0]
    assert primary["cute_vector_widths"] == [1, vector]


def test_new_family_preserves_all_previous_seeds_and_default_order():
    prior = tuple(
        heuristic
        for heuristic in HEURISTICS_BY_BACKEND["cute"]
        if heuristic is not CuteBoundedLoopCacheHeuristic
    )
    with _cpu_target():
        with patch.dict(HEURISTICS_BY_BACKEND, {"cute": prior}):
            old = _bound(shape=(32, 4096))
        new = _bound(shape=(32, 4096))
        old_seeds = old.config_spec.compiler_seed_configs
        assert new.config_spec.compiler_seed_configs[: len(old_seeds)] == old_seeds
        with old.env:
            generation = ConfigGeneration(old.config_spec)
            old_default = generation.unflatten(generation.default_flat())
        with new.env:
            generation = ConfigGeneration(new.config_spec)
            assert generation.unflatten(generation.default_flat()) == old_default
            assert not CuteBoundedLoopCacheHeuristic.should_promote(new.env)
        assert old.to_code(old_default) == new.to_code(old_default)


@pytest.mark.parametrize("count", [2, 7, 30, 100])
def test_bounded_family_preserves_budget_and_explicit_seed_priority(count):
    with _cpu_target():
        bound = _bound(shape=(32, 4096), effort="full")
        explicit = _seeds(bound)[-1]
        population = _population(bound, count, explicit)
        with bound.env:
            generation = ConfigGeneration(bound.config_spec)
            expected = generation.canonicalize_flat(generation.flatten(explicit))[1]
    assert population[1] == expected


def row_energy(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    columns = hl.register_block_size(n)
    for row in hl.tile(m):
        total = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(n, block_size=columns):
            value = x[row, column].to(torch.float32)
            total += torch.sum(value * value, dim=1)
        for column in hl.tile(n, block_size=columns):
            value = x[row, column].to(torch.float32)
            out[row, column] = value * total[:, None]
    return out


def inplace_rescale(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    columns = hl.register_block_size(n)
    for row in hl.tile(m):
        total = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(n, block_size=columns):
            total += torch.sum(x[row, column].to(torch.float32), dim=1)
        for column in hl.tile(n, block_size=columns):
            x[row, column] = x[row, column].to(torch.float32) * total[:, None]
    return x


def _generic_bound(function, dtype):
    kernel = helion.kernel(
        function, backend="cute", static_shapes=False, autotune_effort="none"
    )
    return kernel._bind_isolated((torch.empty((5, 1025), dtype=dtype),))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_general_single_reduction_and_later_raw_reload_use_same_proof(dtype):
    with _cpu_target():
        bound = _generic_bound(row_energy, dtype)
        primary = _seeds(bound)[0]
        source = bound.to_code(primary)
        original = bound.to_code(
            helion.Config.from_dict(
                {**primary.config, "cute_reduction_sequence": "scalar"}
            )
        )
    functions = _functions(source)
    assert ast.dump(functions["_helion_row_energy"]) == ast.dump(
        _functions(original)["_helion_row_energy"]
    )
    fast = ast.unparse(functions["_helion_row_energy_bounded"])
    assert "cute.make_rmem_tensor" in fast
    assert "cute.arch.load(" in fast
    assert "n = cutlass.Int64(cute.size(x, mode=[1]))" in fast


def test_inplace_output_declines_cache_and_keeps_original_kernel():
    with _cpu_target():
        bound = _generic_bound(inplace_rescale, torch.bfloat16)
        primary = _seeds(bound)[0]
        source = bound.to_code(primary)
        original = bound.to_code(
            helion.Config.from_dict(
                {**primary.config, "cute_reduction_sequence": "scalar"}
            )
        )
    assert source == original
    assert "_helion_inplace_rescale_bounded" not in source


def test_unsupported_raw_dtype_does_not_borrow_promoted_fp32_width():
    with _cpu_target():
        bound = _generic_bound(row_energy, torch.int32)
        with bound.env:
            assert (
                CuteBoundedLoopCacheHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
                is None
            )
