from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.jagged_layer_norm import jagged_layer_norm_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.backend import CuteBackend
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.config_spec import NumThreadsSpec

if TYPE_CHECKING:
    from helion._compiler.cute.loop_nesting import TileLoopPath


def _spec(sizes: list[int], paths: tuple[TileLoopPath, ...]) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
        device=torch.device("cpu"),
        num_sm=148,
    )
    for block_id, size in enumerate(sizes):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=size))
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    for group in dict.fromkeys(group for path in paths for group in path):
        if len(group) > 1:
            spec.loop_orders.append(LoopOrderSpec(list(group)))
    spec.cute_tile_loop_paths = paths
    return spec


def _round_trip(
    spec: ConfigSpec, sizes: list[int], threads: list[int]
) -> helion.Config:
    generation = ConfigGeneration(spec)
    return generation.unflatten(
        generation.flatten(helion.Config(block_sizes=sizes, num_threads=threads))
    )


@skipUnlessBackends(["cute"])
def test_registered_jagged_seeds_retain_shared_512_thread_launch() -> None:
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_flatten_nested_reductions=True,
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = _cpu_bind(
            kernel,
            (torch.empty((32641, 128)), torch.empty(257, dtype=torch.int64), 1e-6),
        )
        spec = bound.config_spec
        assert spec.cute_tile_loop_paths == (
            ((0,), (1,)),
            ((0,), (2,)),
            ((0,), (3,)),
        )
        generation = ConfigGeneration(spec)
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.block_sizes == [1, 32768, 32768, 32768]
            and seed.num_threads == [1, 512, 512, 512]
        ]
        assert seeds
        for seed in seeds:
            repaired = generation.unflatten(generation.flatten(seed))
            assert repaired.block_sizes == seed.block_sizes
            assert repaired.num_threads == seed.num_threads
            with patch(
                "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
                return_value=232448,
            ):
                assert "block=(512, 1, 1)" in bound.to_code(repaired)


@pytest.mark.parametrize("automatic", (False, True))
def test_nested_axes_still_fit_thread_limit(automatic: bool) -> None:
    sizes = [8, 128, 256]
    spec = _spec(sizes, (((0,), (1,), (2,)),))
    config = _round_trip(spec, sizes, [0, 0, 0] if automatic else sizes)
    extents = [
        thread or size for thread, size in zip(config.num_threads, sizes, strict=True)
    ]
    assert math.prod(extents) <= 1024
    for size, thread in zip(sizes, extents, strict=True):
        assert size % thread == 0


def test_separate_roots_budget_maximum_of_each_axis() -> None:
    sizes = [512, 2, 2, 512]
    spec = _spec(sizes, (((0, 1),), ((2, 3),)))
    repaired = _round_trip(spec, sizes, sizes)
    a, b, c, d = repaired.num_threads
    assert max(a, c) * max(b, d) <= 1024
    assert repaired.num_threads != sizes


def test_loop_order_aligns_separate_root_axes() -> None:
    sizes = [512, 2, 2, 512]
    spec = _spec(sizes, (((0, 1),), ((2, 3),)))
    generation = ConfigGeneration(spec)
    config = helion.Config(
        block_sizes=sizes,
        num_threads=sizes,
        loop_orders=[[0, 1], [1, 0]],
    )
    repaired = generation.unflatten(generation.flatten(config))
    assert repaired.num_threads == sizes
    assert repaired.loop_orders == config.loop_orders


@pytest.mark.parametrize("inactive_kind", ("boundary", "reduction", "unreachable"))
def test_inactive_axes_do_not_shrink_live_siblings(inactive_kind: str) -> None:
    sizes = [1024, 512, 512]
    paths = (
        (((1,),), ((2,),))
        if inactive_kind == "unreachable"
        else (((0,), (1,)), ((0,), (2,)))
    )
    spec = _spec(sizes, paths)
    if inactive_kind == "boundary":
        spec.cute_inactive_tile_block_ids = {0}
    elif inactive_kind == "reduction":
        spec.reduction_block_ids = {0}
    assert _round_trip(spec, sizes, sizes).num_threads == sizes


def test_repeated_block_in_path_owns_one_axis() -> None:
    sizes = [2, 512]
    spec = _spec(sizes, (((0,), (1,), (1,)),))
    assert _round_trip(spec, sizes, sizes).num_threads == sizes


def test_specs_without_topology_keep_existing_budget() -> None:
    sizes = [512, 512, 512]
    spec = _spec(sizes, ())
    assert math.prod(_round_trip(spec, sizes, sizes).num_threads) <= 1024
