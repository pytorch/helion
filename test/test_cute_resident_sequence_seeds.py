from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.softmax import softmax_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics.cute_resident_sequence import (
    CuteResidentSequenceHeuristic,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(
    rows: int, columns: int, *, static: bool = True, flags: bool = True
) -> tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]]:
    kernel = helion.kernel(
        softmax_bwd.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
        cute_region_fission=flags,
        cute_materialize_transformed_operands=flags,
        cute_full_slice_matmul_tiling=flags,
        cute_segmented_matmul_tiling=flags,
        cute_flatten_nested_reductions=flags,
    )
    args = (
        torch.empty((rows, columns), dtype=torch.float16),
        torch.empty((rows, columns), dtype=torch.float16),
    )
    return _cpu_bind(kernel, args), args


def _seeds(bound: BoundKernel) -> list[helion.Config]:
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteResidentSequenceHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert seeds
    return seeds


def _initial_population(
    bound: BoundKernel,
    args: tuple[torch.Tensor, torch.Tensor],
    *,
    count: int | None = None,
    explicit: helion.Config | None = None,
) -> list[helion.Config]:
    profile = get_effort_profile("full").pattern_search
    assert profile is not None
    with (
        bound.env,
        patch.object(
            bound.settings, "autotune_seed_configs", [explicit] if explicit else []
        ),
    ):
        search = PatternSearch(
            bound,
            args,
            initial_population=profile.initial_population if count is None else count,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        flats = search._generate_initial_population_flat()
        configs = [search.config_gen.unflatten(flat) for flat in flats]
        assert [search.config_gen.flatten(config) for config in configs] == flats
        assert configs[0] == search.config_gen.unflatten(
            search.config_gen.default_flat()
        )
    return configs


@pytest.fixture(scope="module", params=[(256, 1024), (4096, 4096), (4096, 32768)])
def study_case(
    request: pytest.FixtureRequest,
) -> tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]]:
    rows, columns = request.param
    return _bind(rows, columns)


def test_full_random_population_reaches_wide_reduction_threads(
    study_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, args = study_case
    seeds = _seeds(bound)
    population = _initial_population(bound, args)
    assert len(population) == 100
    expected_threads = {512} if args[0].size(1) == 1024 else {512, 1024}
    added = [seed for seed in seeds if seed.num_threads[1] in expected_threads]
    assert {
        (seed.num_threads[1], seed["cute_reduction_sequence"]) for seed in added
    } == {
        (threads, schedule)
        for threads in expected_threads
        for schedule in ("resident", "reload")
    }
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        for seed in added:
            flat, normalized = generation.canonicalize_flat(generation.flatten(seed))
            assert normalized in population
            assert generation.unflatten(flat) == normalized
            assert normalized.num_threads == seed.num_threads
            for key in (
                "block_sizes",
                "cute_vector_widths",
                "cute_lane_layouts",
                "cute_reduction_sequence",
                "cute_cluster_n",
            ):
                assert normalized[key] == seed[key]
            assert seed["pid_type"] == normalized.config.get("pid_type", "flat")


def test_wide_reload_seed_finishes_one_reduction_per_original_tile(
    study_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, _args = study_case
    for seed in _seeds(bound):
        if seed.num_threads[1] < 512 or seed["cute_reduction_sequence"] != "reload":
            continue
        source = bound.to_code(seed)
        assert f"block=({seed.num_threads[1]}, 1, 1)" in source
        assert "sequence_acc" in source
        assert "sequence_values" not in source
        assert "sum_1 = cutlass.Float16(" in source
        assert "v_1 = cutlass.Float32(sum_1)" in source
        tree = ast.parse(source)
        reductions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.warp_reduction_sum"
        ]
        assert len(reductions) == 1
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id.startswith(("lane_", "vec_lane_"))
            ):
                assert not any(reduction in ast.walk(node) for reduction in reductions)


def test_existing_seed_prefix_and_primary_remain_stable(
    study_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, _args = study_case
    seeds = _seeds(bound)
    legacy = [seed for seed in seeds if seed.num_threads[1] <= 256]
    assert seeds[: len(legacy)] == legacy
    assert bound.host_function is not None
    with bound.env:
        assert (
            CuteResidentSequenceHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
            == legacy[0]
        )
        assert not CuteResidentSequenceHeuristic.should_promote(bound.env)
        generation = ConfigGeneration(bound.config_spec)
        default = generation.unflatten(generation.default_flat())
    before = bound.to_code(default)
    additions = set(seeds) - set(legacy)
    with patch.object(
        bound.config_spec,
        "compiler_seed_configs",
        [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed not in additions
        ],
    ):
        with bound.env:
            old_generation = ConfigGeneration(bound.config_spec)
            assert old_generation.unflatten(old_generation.default_flat()) == default
        assert bound.to_code(default) == before


@pytest.mark.parametrize("count", [2, 7, 30, 100])
def test_new_seeds_keep_population_budget_and_user_priority(count: int) -> None:
    bound, args = _bind(32, 4096)
    explicit = _seeds(bound)[-1]
    population = _initial_population(bound, args, count=count, explicit=explicit)
    assert len(population) == count
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _flat, expected = generation.canonicalize_flat(generation.flatten(explicit))
    assert population[1] == expected


@pytest.mark.parametrize("static,flags", [(False, False), (False, True), (True, False)])
def test_wide_seeds_do_not_depend_on_static_shapes_or_optional_matmul_passes(
    static: bool, flags: bool
) -> None:
    bound, args = _bind(17, 8193, static=static, flags=flags)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
    population = _initial_population(bound, args)
    for seed in _seeds(bound):
        if seed.num_threads[1] not in {512, 1024}:
            continue
        _flat, normalized = generation.canonicalize_flat(generation.flatten(seed))
        assert normalized in population
        source = bound.to_code(normalized)
        assert "sequence_acc" in source
        assert "sum_1 = cutlass.Float16(" in source


@pytest.mark.parametrize(
    "columns,expected_threads", [(512, set()), (1024, {512}), (2048, {512, 1024})]
)
def test_wide_seeds_keep_owned_lane_and_tile_constraints(
    columns: int, expected_threads: set[int]
) -> None:
    bound, _args = _bind(8, columns)
    seeds = _seeds(bound)
    added = [seed for seed in seeds if seed.num_threads[1] >= 512]
    assert {seed.num_threads[1] for seed in added} == expected_threads
    for seed in added:
        vector = cast("list[int]", seed["cute_vector_widths"])[1]
        assert seed.block_sizes[1] >= 2 * seed.num_threads[1] * vector
        assert seed.block_sizes[1] // seed.num_threads[1] <= 128
        assert seed.num_threads[0] * seed.num_threads[1] <= 1024
    with patch.object(bound.config_spec.block_sizes[1], "max_size", columns // 2):
        assert bound.host_function is not None
        with bound.env:
            assert (
                CuteResidentSequenceHeuristic.get_seed_configs(
                    bound.env, bound.host_function.device_ir
                )
                is None
            )


def test_full_population_reaches_two_level_cache_reload_seed() -> None:
    bound, args = _bind(4096, 4096)
    expected = ["l1_l2_last", "l1_l2_last", "l1_l2_first", "l1_l2_first"]
    selected = [
        seed
        for seed in _seeds(bound)
        if seed.num_threads[1:] == [256, 256]
        and seed.config.get("load_eviction_policies") == expected
    ]
    assert len(selected) == 1
    population = _initial_population(bound, args)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _, normalized = generation.canonicalize_flat(generation.flatten(selected[0]))
    assert normalized in population
    code = bound.to_code(normalized)
    assert "sequence_acc" in code
    assert "_cute_load_l1_l2_evict_last" in code
    assert "_cute_load_l1_l2_evict_first" in code
