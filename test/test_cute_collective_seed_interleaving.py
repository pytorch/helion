from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import checked_initial_population
from test.test_cute_atomic_output_promotions import _assert_fp32_atomic_output
from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_collective_chunk_seeds import _matmul_with_unrelated_knob
from test.test_cute_collective_chunk_seeds import _partitioned_matmul

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.pattern_search import PatternSearch
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda_init() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.fixture(scope="module")
def partitioned_case() -> tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]]:
    args = (
        torch.empty((128, 32768), dtype=torch.float16),
        torch.empty((32768, 256), dtype=torch.float16),
    )
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    return _cpu_bind(kernel, args), args


def _legacy_order(
    spec: ConfigSpec, seeds: list[helion.Config], parameters: tuple[str, ...]
) -> list[helion.Config]:
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
    args: tuple[torch.Tensor, torch.Tensor],
    *,
    seeds: list[helion.Config] | None = None,
    count: int | None = None,
    user_seeds: list[helion.Config] | None = None,
) -> list[helion.Config]:
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
            initial_population=profile.initial_population if count is None else count,
            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
        )
        flat = checked_initial_population(search)
        result = [search.config_gen.unflatten(value) for value in flat]
        assert [search.config_gen.flatten(config) for config in result] == flat
        assert result[0] == search.config_gen.unflatten(
            search.config_gen.default_flat()
        )
        assert len(result) == len(set(result))
        return result


def _schedule(config: helion.Config) -> tuple[object, ...]:
    return (
        config.get("cute_collective_compute"),
        config.get("cute_collective_copy"),
        config.get("cute_collective_stages"),
    )


def test_full_random_population_reaches_coupled_pipeline_chunk_geometries(
    partitioned_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, args = partitioned_case
    legacy = _initial_population(bound, args, seeds=_legacy_registered_seeds(bound))
    actual = _initial_population(bound, args)
    assert len(actual) == len(legacy)
    assert actual == _initial_population(bound, args)
    for stages in (2, 4):
        before = [
            seed for seed in legacy if seed.get("cute_collective_stages") == stages
        ]
        after = [
            seed for seed in actual if seed.get("cute_collective_stages") == stages
        ]
        assert {seed["split_k"] for seed in before} == {1}
        assert {seed["split_k"] for seed in after} == {1, 2, 4, 8, 16, 32, 64, 128, 256}
        for partitions in (16, 64, 256):
            geometries = {
                tuple(seed.block_sizes)
                for seed in after
                if seed["split_k"] == partitions
            }
            assert len(geometries) >= 2
    assert {_schedule(seed) for seed in actual if seed.get("cute_collective_mma")} == {
        _schedule(seed) for seed in legacy if seed.get("cute_collective_mma")
    }
    assert not actual[0].get("cute_collective_mma", False)
    assert any(seed.get("cute_collective_compute") == "tcgen05" for seed in actual)


def test_interleaving_preserves_the_full_seed_set_and_primary(
    partitioned_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, _args = partitioned_case
    original = _legacy_registered_seeds(bound)
    reordered = bound.config_spec.compiler_seed_configs
    assert len(original) == len(reordered)
    assert set(original) == set(reordered)
    assert original[0] == reordered[0]
    assert original != reordered


def test_newly_reached_pipeline_keeps_fp32_atomic_storage(
    partitioned_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    bound, args = partitioned_case
    population = _initial_population(bound, args)
    choice = next(
        seed
        for seed in population
        if seed.block_sizes == [64, 64, 64]
        and seed.get("cute_collective_stages") == 2
        and seed["split_k"] == 64
    )
    code = bound.to_code(choice)
    _assert_fp32_atomic_output(code, "x")
    assert "cute.gemm(" in code
    assert choice.num_threads == [2, 64, 1]


@pytest.mark.parametrize("count", [2, 7, 30, 100])
def test_interleaving_preserves_budget_and_explicit_user_seed_priority(
    partitioned_case: tuple[BoundKernel, tuple[torch.Tensor, torch.Tensor]],
    count: int,
) -> None:
    bound, args = partitioned_case
    explicit = bound.config_spec.compiler_seed_configs[-1]
    population = _initial_population(bound, args, count=count, user_seeds=[explicit])
    # The helper checks exactly count legacy rows plus declared coverage only.
    with bound.env:
        normalized = (
            bound.config_spec.create_config_generation().seed_flat_config_pairs()
        )
        assert population[1] == normalized[-1][1]


@pytest.mark.parametrize(
    "kernel,parameter",
    [(_partitioned_matmul, "partitions"), (_direct_chunk_matmul, "tile_extent")],
)
def test_initial_population_uses_config_provenance_not_parameter_spelling(
    kernel: helion.Kernel, parameter: str
) -> None:
    args = (
        torch.empty((128, 16384), dtype=torch.float16),
        torch.empty((16384, 256), dtype=torch.float16),
    )
    bound = _cpu_bind(kernel, args)
    population = _initial_population(bound, args)
    for stages in (2, 4):
        coupled = []
        for seed in population:
            value = seed[parameter]
            assert isinstance(value, int)
            if seed.get("cute_collective_stages") == stages and value > 1:
                coupled.append(seed)
        assert len({seed[parameter] for seed in coupled}) >= 2
        assert len({tuple(seed.block_sizes) for seed in coupled}) >= 2
        for seed in coupled:
            value = seed[parameter]
            assert isinstance(value, int)
            chunk = (
                value
                if parameter == "tile_extent"
                else helion.next_power_of_2(helion.cdiv(args[0].size(1), value))
            )
            assert chunk >= seed.block_sizes[2]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordinary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile([m, n]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        out[row, column] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fixed_chunk_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for row, column, outer in hl.tile([m, n, k], block_size=[None, None, 64]):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(outer.begin, outer.end):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        hl.atomic_add(out, [row, column], acc)
    return out


@pytest.mark.parametrize(
    "kernel,static_shapes",
    [
        (_ordinary_matmul, True),
        (_fixed_chunk_matmul, True),
        (matmul_split_k, False),
    ],
)
def test_no_config_owned_k_chunk_preserves_initial_population(
    kernel: helion.Kernel, static_shapes: bool
) -> None:
    args = (
        torch.empty((128, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
    )
    variant = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
    )
    bound = _cpu_bind(variant, args)
    legacy_seeds = _legacy_registered_seeds(bound)
    assert legacy_seeds == bound.config_spec.compiler_seed_configs
    assert _initial_population(bound, args, seeds=legacy_seeds) == _initial_population(
        bound, args
    )


def test_unrelated_knob_does_not_group_computed_operand_schedules() -> None:
    args = (
        torch.empty((128, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
    )
    bound = _cpu_bind(_matmul_with_unrelated_knob, args)
    assert bound.host_function is not None
    # This checks this heuristic's permutation, before later independent
    # overlays append their own policy siblings to the merged registry.
    with bound.env:
        actual = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        with patch.object(
            CuteCollectiveMatmulHeuristic,
            "_interleave_bounded_k_seeds",
            side_effect=_legacy_order,
        ):
            legacy = CuteCollectiveMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
    assert any(seed.get("cute_collective_tmem_a") for seed in legacy)
    assert len(legacy) == len(actual)
    assert set(legacy) == set(actual)
    assert legacy[0] == actual[0]
    assert legacy != actual
    assert all("gain" not in seed.config for seed in actual)
    # This kernel's computed A independently admits the TMEM schedule family.
    # Its unrelated gain remains outside both K provenance and grouping keys.
    with_gain = [
        helion.Config.from_dict(seed.config | {"gain": 1 << (index % 3)})
        for index, seed in enumerate(legacy)
    ]
    interleaved = CuteCollectiveMatmulHeuristic._interleave_bounded_k_seeds(
        bound.config_spec, with_gain, ()
    )
    assert [
        helion.Config.from_dict(
            {key: value for key, value in seed.config.items() if key != "gain"}
        )
        for seed in interleaved
    ] == actual
