from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.backend import CuteBackend
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
