from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from .test_cute_grid_launch_extents import _launch_block
from .test_cute_grid_launch_extents import _merged_copy
from .test_cute_grid_launch_extents import _mixed_rank_copy
from .test_cute_grid_launch_extents import _opposed_grids
import helion
from helion._compiler.autotuner_heuristics.cute import CutePointwiseVecHeuristic
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import RootGridFact
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _leading_fixed_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for outer, row, col in hl.tile(x.shape, block_size=[1, None, None]):
        out[outer, row, col] = x[outer, row, col] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_grid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _plain_grids(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(y)
    for tile in hl.tile(x.shape):
        first[tile] = x[tile] + 1
    for tile in hl.tile(y.shape):
        second[tile] = y[tile] + 1
    return first, second


@contextmanager
def _bound(kernel: Any, args: tuple[Any, ...]):
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
    ):
        yield kernel._bind_isolated(args)


def _roundtrip(bound: Any, config: helion.Config) -> helion.Config:
    gen = ConfigGeneration(bound.config_spec)
    return gen.unflatten(gen.flatten(config))


@pytest.mark.parametrize("threads", [(128, 128), (256, 128), (128, 256)])
@pytest.mark.parametrize("vector", [1, 8])
def test_independent_mixed_rank_threads_roundtrip(
    threads: tuple[int, int], vector: int
) -> None:
    args = (torch.empty((2, 4096)), torch.empty((1024,)), torch.empty((1024,)))
    config = helion.Config(
        block_sizes=[1024, 256],
        num_threads=list(threads),
        cute_vector_widths=[1, vector, 1],
        cute_lane_layouts=["blocked", "strided", "blocked"],
    )
    with _bound(_mixed_rank_copy, args) as bound:
        result = _roundtrip(bound, config)
        assert result.num_threads == list(threads)
        assert result.block_sizes == config.block_sizes
        assert result.config["cute_vector_widths"] == [1, vector, 1]
        assert _launch_block(bound.to_code(result)) == (1, max(threads), 1)


@pytest.mark.parametrize("flatten", [False, True])
def test_simultaneous_axes_still_repair_thread_budget(flatten: bool) -> None:
    config = helion.Config(
        block_sizes=[128, 128], num_threads=[128, 128], flatten_loops=[flatten]
    )
    with _bound(_plain_grid, (torch.empty((256, 256)),)) as bound:
        assert len(bound.config_spec.flatten_loops) == 1
        result = _roundtrip(bound, config)
        assert result.num_threads == [32, 32]
        assert math.prod(_launch_block(bound.to_code(result))) <= 1024


def test_missing_grid_topology_retains_conservative_repair() -> None:
    with _bound(_merged_copy, (torch.empty(2048), torch.empty(2048))) as bound:
        # Tile-loop paths normally budget separate roots by physical axis.
        # Without them, the grid facts are the only topology; without those
        # too, the conservative product repair must still apply.
        bound.config_spec.cute_tile_loop_paths = ()
        bound.config_spec.kernel_grid_fact = None
        result = _roundtrip(
            bound, helion.Config(block_sizes=[1024, 1024], num_threads=[128, 128])
        )
        assert result.num_threads == [32, 32]


def test_nonroot_thread_axis_remains_in_every_group() -> None:
    with _bound(_opposed_grids, (torch.empty((256, 256)),) * 2) as bound:
        # Exercise the grid-fact group repair that backs kernels without
        # tile-loop paths. Axis1 has no root-owner proof there; it must
        # coexist with either root.
        bound.config_spec.cute_tile_loop_paths = ()
        bound.config_spec.kernel_grid_fact = KernelGridFact(
            roots=(RootGridFact(0, (0,)), RootGridFact(1, (2, 3))),
            graph_to_root=(),
        )
        gen = ConfigGeneration(bound.config_spec)
        groups = gen._cute_coexisting_thread_groups(gen.num_threads_indices)
        assert groups == [gen.num_threads_indices[:2], gen.num_threads_indices[1:]]
        result = gen.unflatten(
            gen.flatten(helion.Config(block_sizes=[128] * 4, num_threads=[128] * 4))
        )
        assert math.prod(result.num_threads[:2]) <= 1024
        assert math.prod(result.num_threads[1:]) <= 1024


def test_aliased_thread_axis_belongs_to_both_root_groups() -> None:
    with _bound(_opposed_grids, (torch.empty((256, 256)),) * 2) as bound:
        spec = bound.config_spec
        spec.num_threads[0].block_ids.append(99)
        spec.kernel_grid_fact = KernelGridFact(
            roots=(RootGridFact(0, (0, 1)), RootGridFact(1, (99, 2, 3))),
            graph_to_root=(),
        )
        gen = ConfigGeneration(spec)
        indices = gen.num_threads_indices
        assert gen._cute_coexisting_thread_groups(indices) == [
            indices[:2],
            [indices[0], *indices[2:]],
        ]


@pytest.mark.parametrize("opposed", [False, True])
def test_flattened_root_auto_threads_are_repaired_per_root(opposed: bool) -> None:
    with _bound(_plain_grids, (torch.empty((256, 256)),) * 2) as bound:
        assert len(bound.config_spec.flatten_loops) == 2
        result = _roundtrip(
            bound,
            helion.Config(
                block_sizes=[128] * 4,
                num_threads=[128, 0, 0, 128] if opposed else [128, 0, 128, 0],
                flatten_loops=[True, True],
            ),
        )
        if opposed:
            # The launch takes each physical axis's maximum across roots, so
            # both roots shrink until max(32, 8) * max(8, 32) fits the budget.
            assert result.num_threads == [32, 8, 8, 32]
        else:
            assert result.num_threads == [128, 8, 128, 8]
            assert math.prod(result.num_threads[:2]) == 1024
            assert math.prod(result.num_threads[2:]) == 1024
        assert math.prod(_launch_block(bound.to_code(result))) <= 1024


def test_distinct_physical_root_axes_are_repaired_to_fit() -> None:
    config = helion.Config(
        block_sizes=[16, 2048, 2048, 16],
        num_threads=[16, 64, 64, 16],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["strided"] * 4,
    )
    args = (torch.empty((32, 4096)), torch.empty((4096, 32)))
    with _bound(_opposed_grids, args) as bound:
        # Explicit counts on opposed roots would launch max(16, 64) * max(64,
        # 16) threads. The tile-loop path repair shrinks the wide axes instead
        # of leaving a config that codegen must reject.
        result = _roundtrip(bound, config)
        assert result.num_threads == [16, 32, 32, 16]
        assert math.prod(_launch_block(bound.to_code(result))) <= 1024


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 8), (torch.float32, 4)])
def test_multigrid_pointwise_seeds_transfer_by_live_block_id(
    dtype: torch.dtype, width: int
) -> None:
    args = (
        torch.empty((2, 4096), dtype=dtype),
        torch.empty((1024,), dtype=dtype),
        torch.empty((1024,), dtype=dtype),
    )
    with _bound(_mixed_rank_copy, args) as bound:
        spec = bound.config_spec
        assert len(spec.block_sizes) == len(spec.num_threads) == 2
        assert len(spec.cute_vector_widths) == 3
        generator = ConfigGeneration(spec)
        failures: list[str] = []
        seeds = [
            config for _, config in generator.seed_flat_config_pairs(failures.append)
        ]
        assert not failures
        assert seeds
        assert len(seeds) <= 24
        target = next(
            config
            for config in seeds
            if config.block_sizes == [128 * width, 128]
            and config.num_threads == [128, 128]
            and config.config["cute_vector_widths"] == [1, width, 1]
        )
        assert _launch_block(bound.to_code(target)) == (1, 128, 1)


def test_single_root_extra_vector_slot_seed_transfer() -> None:
    with _bound(
        _leading_fixed_copy, (torch.empty((2, 32, 2048), dtype=torch.bfloat16),)
    ) as bound:
        spec = bound.config_spec
        assert len(spec.block_sizes) == 2
        assert len(spec.cute_vector_widths) == 3
        failures: list[str] = []
        seeds = ConfigGeneration(spec).seed_flat_config_pairs(failures.append)
        assert seeds
        assert not failures
        assert all(
            len(cast("list[int]", config.config["cute_vector_widths"])) == 3
            for _, config in seeds
        )


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 8), (torch.float32, 4)])
def test_single_root_pointwise_vector_seed_is_unchanged(
    dtype: torch.dtype, width: int
) -> None:
    with _bound(_plain_copy, (torch.empty((8192,), dtype=dtype),)) as bound:
        with bound.env:
            seed = CutePointwiseVecHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
        assert seed is not None
        assert seed.block_sizes == [256 * width * 2]
        assert seed.num_threads == [256]
        assert seed.config["cute_vector_widths"] == [width]


def test_byte_storage_vector_seed_uses_legal_fragment_choice() -> None:
    with _bound(_plain_copy, (torch.empty((8192,), dtype=torch.bool),)) as bound:
        failures: list[str] = []
        seeds = ConfigGeneration(bound.config_spec).seed_flat_config_pairs(
            failures.append
        )
        assert seeds
        assert not failures
        assert all(
            max(cast("list[int]", config.config["cute_vector_widths"])) <= 8
            for _, config in seeds
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("threads", [(128, 128), (256, 128), (128, 256)])
@pytest.mark.parametrize("vector", [1, 8])
def test_multigrid_search_roundtrip_runtime(
    threads: tuple[int, int], vector: int
) -> None:
    x = torch.arange(8192, device=DEVICE, dtype=torch.float32).reshape(2, 4096)
    y = torch.arange(1024, device=DEVICE, dtype=torch.float32)
    storage = torch.full((1280,), -98765.0, device=DEVICE)
    second = storage[128:1152]
    args = (x, y, second)
    saved = (x.clone(), y.clone())
    bound = _mixed_rank_copy._bind_isolated(args)
    config = _roundtrip(
        bound,
        helion.Config(
            block_sizes=[1024, 256],
            num_threads=list(threads),
            cute_vector_widths=[1, vector, 1],
            cute_lane_layouts=["blocked", "strided", "blocked"],
        ),
    )
    assert config.num_threads == list(threads)
    run = bound.compile_config(config)
    expected = x + torch.arange(2, device=DEVICE)[:, None]
    torch.testing.assert_close(run(*args), expected, atol=0, rtol=0)
    torch.testing.assert_close(run(*args), expected, atol=0, rtol=0)
    torch.testing.assert_close(second, y, atol=0, rtol=0)
    torch.testing.assert_close((x, y), saved, atol=0, rtol=0)
    assert torch.all(storage[:128] == -98765.0)
    assert torch.all(storage[1152:] == -98765.0)
