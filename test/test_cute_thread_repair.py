from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.exc import InvalidConfig
from helion.exc import NestedDeviceLoopsConflict
import helion.language as hl

pytestmark = pytest.mark.cpu_only

jagged_layer_norm_kernel = import_example("jagged_layer_norm").jagged_layer_norm_kernel

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.cute.loop_nesting import TileLoopPath


def _cute_spec() -> ConfigSpec:
    """An empty CuTe ``ConfigSpec`` for a B200 (sm_100, 148 SMs)."""
    return ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 0),
        device=torch.device("cpu"),
        num_sm=148,
    )


def _spec(sizes: list[int], paths: tuple[TileLoopPath, ...]) -> ConfigSpec:
    spec = _cute_spec()
    for block_id, size in enumerate(sizes):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=size))
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    for group in dict.fromkeys(group for path in paths for group in path):
        if len(group) > 1:
            spec.loop_orders.append(LoopOrderSpec(list(group)))
    # As a compile does: tiles that start with one block share its spec.
    spec._remove_duplicates()
    spec.cute_tile_loop_path_reductions = dict.fromkeys(paths, frozenset())
    return spec


def _round_trip(
    spec: ConfigSpec, sizes: list[int], threads: list[int]
) -> helion.Config:
    generation = ConfigGeneration(spec)
    return generation.unflatten(
        generation.flatten(helion.Config(block_sizes=sizes, num_threads=threads))
    )


def test_registered_jagged_seeds_retain_shared_512_thread_launch() -> None:
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_flatten_nested_reductions=True,
    )
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        bound = cpu_bind(
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
                "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
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


@pytest.mark.parametrize(
    ("sizes", "paths", "threads", "pack", "fitted"),
    [
        # explicit counts: 1024 threads leave no room for a second tile, 512
        # for two of four, 256 for all four
        ([4, 256], (((9,), (0, 1)),), [4, 256], 2, 1),
        ([4, 256], (((9,), (0, 1)),), [4, 128], 4, 2),
        ([4, 256], (((9,), (0, 1)),), [4, 64], 4, 4),
        # auto counts shrink into what the pack leaves of a block
        ([4, 256], (((9,), (0, 1)),), [0, 0], 2, 2),
        # the tile's auto axis runs on one thread beside the packed axis
        ([2, 16, 32], (((9,), (0, 1, 2)),), [0, 16, 32], 8, 2),
        # the third explicit axis runs as a lane loop
        ([8, 4, 32], (((9,), (0, 1, 2)),), [8, 4, 32], 4, 4),
        # an enclosing tile's auto axis keeps its threads
        ([128, 4, 2], (((0, 9), (1, 2)),), [0, 4, 2], 8, 2),
        # sibling tiles each run their own auto axis on one thread
        (
            [2, 16, 32, 2, 16, 32],
            (((9,), (0, 1, 2)), ((9,), (3, 4, 5))),
            [0, 16, 32, 2, 0, 0],
            8,
            2,
        ),
        # the root tile's auto count shrinks beside the pack
        ([256], (((0, 9),),), [0], 8, 8),
    ],
)
def test_packed_tiles_fit_one_block(
    sizes: list[int],
    paths: tuple[TileLoopPath, ...],
    threads: list[int],
    pack: int,
    fitted: int,
) -> None:
    """Each grid tile ``cute_tiles_per_block`` packs into a CTA (the grid
    axis of block 9) runs on threads of its own: the repair halves the pack
    until the tiles' threads, laid out as the planner lays out each tile,
    times the pack fit in one block, and the tiles keep their threads."""
    spec = _spec(sizes, paths)
    spec.cute_tile_pack_counts = {9: 64}
    spec.cute_inactive_tile_block_ids = {9}
    generation = ConfigGeneration(spec)
    drawn = helion.Config(
        block_sizes=sizes, num_threads=threads, cute_tiles_per_block=pack
    )
    repaired = generation.unflatten(generation.flatten(drawn))
    assert (repaired.num_threads or [0] * len(sizes)) == threads
    assert repaired.config["cute_tiles_per_block"] == fitted


def test_packed_tiles_take_their_axes_before_a_reduction() -> None:
    """Code generation fits a row sum (block 2) into what the packed tiles
    leave of a block, counting their threads on all their axes: eight
    16 x 32 tiles leave it one thread and no launch axis, so the tiles take
    all three axes and need 4096 threads, and the pack halves to 2.  Giving
    the sum the leading axis first would cut the tile to 16 threads and keep
    the pack of 8, which code generation refuses (as it does 4)."""
    spec = _spec([16, 64, 512], (((9,), (0, 1)),))
    spec.cute_tile_pack_counts = {9: 64}
    spec.cute_inactive_tile_block_ids = {9}
    spec.reduction_block_ids = {2}
    generation = ConfigGeneration(spec)
    drawn = helion.Config(
        block_sizes=[16, 64, 512], num_threads=[16, 32, 0], cute_tiles_per_block=8
    )
    repaired = generation.unflatten(generation.flatten(drawn))
    assert repaired.config["cute_tiles_per_block"] == 2


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


def test_a_tile_sharing_a_block_with_another_keeps_its_own_loop_order() -> None:
    # Two roots tile one registered block, as [row, col] (with a loop order)
    # and as [row] alone (with none): the 1-D level took the 2-D tile's order
    # and indexed past its one block.
    sizes = [4, 32]
    spec = _spec(sizes, (((0, 1),), ((0,),)))
    generation = ConfigGeneration(spec)
    config = helion.Config(block_sizes=sizes, num_threads=sizes, loop_orders=[[1, 0]])
    repaired = generation.unflatten(generation.flatten(config))
    assert repaired.num_threads == sizes
    assert repaired.loop_orders == [[1, 0]]


def test_tiles_starting_with_one_block_share_its_loop_order() -> None:
    # [row, col] in one root and [row, k] in another keep one loop-order spec,
    # which codegen applies to both: each launches its 512 threads on x and
    # its 2 on y, 1024 in all.
    sizes = [2, 512, 512]
    spec = _spec(sizes, (((0, 1),), ((0, 2),)))
    assert len(spec.loop_orders) == 1
    generation = ConfigGeneration(spec)
    config = helion.Config(block_sizes=sizes, num_threads=sizes, loop_orders=[[1, 0]])
    repaired = generation.unflatten(generation.flatten(config))
    assert repaired.num_threads == sizes
    assert repaired.loop_orders == [[1, 0]]


def test_tiles_sharing_a_block_anywhere_share_its_loop_order() -> None:
    # [row, col] in one root and [k, row] in another keep [k, row]'s loop-order
    # spec, which codegen applies to both: [1, 0] lays [row, col] out col
    # first, so each root asks for 512 threads on x and 512 on y, and the
    # repair shrinks them.
    sizes = [2, 512, 512]
    spec = _spec(sizes, (((0, 1),), ((2, 0),)))
    assert [list(order.block_ids) for order in spec.loop_orders] == [[2, 0]]
    generation = ConfigGeneration(spec)
    config = helion.Config(block_sizes=sizes, num_threads=sizes, loop_orders=[[1, 0]])
    repaired = generation.unflatten(generation.flatten(config))
    threads = repaired.num_threads
    assert max(threads[1], threads[0]) * max(threads[0], threads[2]) <= 1024
    assert repaired.loop_orders == [[1, 0]]


def _two_roots_sharing_rows(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = helion.language.register_block_size(x.size(0))
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for row, col in helion.language.tile(x.shape, block_size=[rows, None]):
        out_x[row, col] = x[row, col] + 1
    for k, row in helion.language.tile(y.shape, block_size=[None, rows]):
        out_y[k, row] = y[k, row] * 2
    return out_x, out_y


def test_two_roots_sharing_a_block_launch_the_repaired_loop_order() -> None:
    """The same two roots compiled: the repair reads the loop order the
    compile registers for [row, col] (that of [k, row]), as codegen does, so
    [2, 512, 512] threads with [1, 0] shrink to [2, 32, 32], and the kernel
    launches them.  Read by [row, col]'s own first block, the order came out
    the default, the repair kept 512 x 512 threads, and code generation
    refused the launch."""
    kernel = helion.kernel(
        _two_roots_sharing_rows,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(2, 512), torch.empty(512, 2)))
        generation = ConfigGeneration(bound.config_spec)
        config = helion.Config(
            block_sizes=[2, 512, 512], num_threads=[2, 512, 512], loop_orders=[[1, 0]]
        )
        repaired = generation.unflatten(generation.flatten(config))
        assert repaired.num_threads == [2, 32, 32]
        assert repaired.loop_orders == [[1, 0]]
        assert "block=(32, 32, 1)" in bound.to_code(repaired)


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


def _nested_reduction_spec(*, bounded: bool = False) -> ConfigSpec:
    """A ``[64, 384]`` tile around a persistent 128-element reduction, the
    shape of BatchNorm's split-reduction partials; ``bounded`` caps block 1
    by block 0 as ``hl.tile(outer.begin, outer.end)`` does."""
    spec = _cute_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
    spec.block_sizes.append(
        BlockSizeSpec(
            block_id=1, size_hint=384, bounded_by_block_id=0 if bounded else None
        )
    )
    for block_id, size in ((0, 64), (1, 384), (2, 128)):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=2, size_hint=128))
    spec.reduction_block_ids.add(2)
    return spec


def _tile_threads(spec: ConfigSpec, config: helion.Config) -> int:
    """The threads of the tile's axes, each its ``num_threads`` entry when
    set, else its block size, resolved by block id."""
    block_sizes = config.config["block_sizes"]
    num_threads = config.config.get("num_threads") or []
    total = 1
    for block in spec.block_sizes:
        count = spec.num_threads.config_get(num_threads, block.block_id, 0)
        total *= count or spec.block_sizes.config_get(block_sizes, block.block_id)
    return total


def _normalized(spec: ConfigSpec, **values: object) -> helion.Config:
    config = helion.Config(**values)
    spec.normalize(config, _fix_invalid=True)
    return config


@pytest.mark.parametrize(
    "values",
    [
        # The default block sizes give the tile the whole 1024-thread budget.
        {"block_sizes": [32, 32], "reduction_loops": [None]},
        # An explicit 96-thread seed halves onto powers of two.
        {"block_sizes": [64, 16], "num_threads": [96, 0, 0], "reduction_loops": [None]},
        # A chunk past the extent normalizes to the persistent None.
        {"block_sizes": [32, 32], "reduction_loops": [256]},
    ],
)
def test_generated_tile_leaves_nested_reduction_a_warp(values: dict) -> None:
    spec = _nested_reduction_spec()
    repaired = _normalized(spec, **values)
    assert _tile_threads(spec, repaired) <= 1024 // 32
    for count in repaired.config.get("num_threads") or []:
        assert count == 0 or count & (count - 1) == 0
    # A fixed point of normalize, also without repairs.
    again = helion.Config(**repaired.config)
    spec.normalize(again)
    assert again.config == repaired.config


@pytest.mark.parametrize("num_threads", [None, [0, 32, 0]])
def test_bounded_tile_repair_keeps_inner_within_outer(
    num_threads: list[int] | None,
) -> None:
    """Of an equally wide bounded pair the repair halves the inner block
    first.  Halving the outer one would clamp the inner one too, so the pair
    would give up three quarters of its threads per step and go past the
    warp's share of the budget, from [32, 32] to 16 tile threads."""
    spec = _nested_reduction_spec(bounded=True)
    values: dict[str, object] = {"block_sizes": [32, 32], "reduction_loops": [None]}
    if num_threads is not None:
        values["num_threads"] = num_threads
    repaired = _normalized(spec, **values)
    outer, inner = repaired.config["block_sizes"]
    assert inner <= outer
    assert _tile_threads(spec, repaired) == 1024 // 32
    again = helion.Config(**repaired.config)
    spec.normalize(again, _fix_invalid=True)
    assert again.config == repaired.config


def test_explicit_tile_without_reduction_threads_is_rejected() -> None:
    spec = _nested_reduction_spec()
    with pytest.raises(InvalidConfig, match="no thread budget"):
        spec.normalize(helion.Config(block_sizes=[32, 32], reduction_loops=[None]))


@pytest.mark.parametrize(
    ("size", "indexed", "bounded", "values"),
    [
        # ``normalize`` rolls a persistent reduction past its thread threshold
        # (a warp for argmin/argmax, else the 1024-thread budget), and a rolled
        # reduction is not rejected: its threads scale to what the tile leaves.
        (128, True, False, {"block_sizes": [32, 32], "reduction_loops": [None]}),
        (2048, False, False, {"block_sizes": [32, 32], "reduction_loops": [None]}),
        # A tuple or a scalar is the chunk ``[32]``.
        (128, False, False, {"block_sizes": [32, 32], "reduction_loops": (32,)}),
        (128, False, False, {"block_sizes": [32, 32], "reduction_loops": 32}),
        # The bounded-tile clamp fits [16, 128] to [16, 16], which leaves the
        # reduction four threads.
        (128, False, True, {"block_sizes": [16, 128], "reduction_loops": [None]}),
    ],
)
def test_repair_leaves_a_config_normalize_accepts(
    size: int, indexed: bool, bounded: bool, values: dict
) -> None:
    spec = _nested_reduction_spec(bounded=bounded)
    # A ``size``-element reduction, an argmin/argmax with ``indexed``.
    spec.reduction_loops[0] = ReductionLoopSpec(block_id=2, size_hint=size)
    if indexed:
        spec.cute_indexed_reduction_block_ids.add(2)
    accepted = helion.Config(**values)
    spec.normalize(accepted)
    assert _normalized(spec, **values).config == accepted.config


def test_repair_leaves_a_pinned_reduction_slot_alone() -> None:
    # A pinned slot (a reduction older versions rolled, now persistent) is
    # never rejected, so the tile that leaves it no threads is kept, as in
    # the other configs ``normalize`` accepts.
    spec = _nested_reduction_spec()
    spec.reduction_loops[0] = ReductionLoopSpec(
        block_id=2, size_hint=128, persistent_reason="rolled by older versions"
    )
    values = {"block_sizes": [32, 32], "reduction_loops": [None]}
    accepted = helion.Config(**values)
    spec.normalize(accepted)
    assert accepted.config["block_sizes"] == [32, 32]
    assert _normalized(spec, **values).config == accepted.config


def test_bounded_clamp_fits_each_thread_slot_to_its_own_block() -> None:
    """The bounded-tile clamp fits ``num_threads`` slots to the block sizes
    of their own block ids: here the reduction's slot (block 2) comes first
    and keeps its 128 threads, and the slot of the clamped block 1 fits its
    8 elements."""
    spec = _cute_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
    spec.block_sizes.append(
        BlockSizeSpec(block_id=1, size_hint=128, bounded_by_block_id=0)
    )
    for block_id in (2, 0, 1):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=128))
    values: dict[str, object] = {"block_sizes": [8, 128], "num_threads": [128, 8, 128]}
    spec.clamp_bounded_block_sizes(values)
    assert values == {"block_sizes": [8, 8], "num_threads": [128, 8, 8]}


def test_bounded_clamp_floors_a_slot_to_a_power_of_two() -> None:
    """The clamp refits the ``num_threads`` slot of the block it clamps: a
    whole number of warps that is not a power of two (96 threads) becomes
    the largest power of two within its block, 64.  Halving 96 never divides
    a power of two, so the slot fell to one thread.  The slot of block 2,
    which keeps its size, keeps its 96 threads, as it does when no block is
    clamped."""
    spec = _cute_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
    spec.block_sizes.append(
        BlockSizeSpec(block_id=1, size_hint=256, bounded_by_block_id=0)
    )
    spec.block_sizes.append(BlockSizeSpec(block_id=2, size_hint=128))
    for block_id, size in ((0, 128), (1, 256), (2, 128)):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    values: dict[str, object] = {
        "block_sizes": [128, 256, 128],
        "num_threads": [0, 96, 96],
    }
    spec.clamp_bounded_block_sizes(values)
    assert values == {"block_sizes": [128, 128, 128], "num_threads": [0, 64, 96]}


def test_flat_repair_floors_a_slot_to_a_power_of_two() -> None:
    """The flat config repair fits a 96-thread slot to its 128 block as the
    bounded-tile clamp does, at 64 threads: it fell to one thread."""
    spec = _spec([128, 4], (((0,), (1,)),))
    assert _round_trip(spec, [128, 4], [96, 0]).num_threads == [64, 0]


@pytest.mark.parametrize(
    ("constraint", "block_sizes"), [("min_size", [32, 32]), ("pin", [64, 16])]
)
def test_repair_keeps_a_bounded_block_at_its_minimum_and_pin(
    constraint: str, block_sizes: list[int]
) -> None:
    """Halving the outer block of a bounded pair clamps the inner block too,
    so the repair leaves the outer block alone once that clamp would take the
    inner one below its minimum or off its structural pin (16 here): both
    stop at [16, 16], leaving four reduction threads, where halving the outer
    block anyway reached [4, 4].  The rolled chunk is the flat search space's
    smallest, 8, which the four threads take in turns, so the config flattens
    back to itself: a chunk of 4 came back as 8."""
    spec = _nested_reduction_spec(bounded=True)
    inner = spec.block_sizes.block_id_lookup(1)
    if constraint == "min_size":
        inner.update_min(16)
    else:
        inner.pin_structurally(16, reason="a test pin")
    repaired = _normalized(spec, block_sizes=block_sizes, reduction_loops=[None])
    assert repaired.config["block_sizes"] == [16, 16]
    assert repaired.config["reduction_loops"] == [8]
    again = helion.Config(**repaired.config)
    spec.normalize(again, _fix_invalid=True)
    assert again.config == repaired.config
    generation = ConfigGeneration(spec)
    flat = generation.unflatten(generation.flatten(repaired))
    spec.normalize(flat, _fix_invalid=True)
    keys = ("block_sizes", "reduction_loops")
    assert {key: flat.config[key] for key in keys} == {
        key: repaired.config[key] for key in keys
    }


def test_repair_halves_a_whole_warp_count_to_the_power_of_two_below() -> None:
    """A 96-thread slot halves to 64, the largest power of two below it: the
    tile of 96 x 8 threads around a 2-element persistent reduction then takes
    the 512 the reduction leaves, where 32 x 8 halved too far."""
    spec = _cute_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
    spec.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=8))
    for block_id, size in ((0, 128), (1, 8), (2, 2)):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=2, size_hint=2))
    spec.reduction_block_ids.add(2)
    repaired = _normalized(
        spec, block_sizes=[128, 8], num_threads=[96, 0, 0], reduction_loops=[None]
    )
    assert repaired.config["num_threads"][0] == 64


def test_a_rolled_chunk_is_a_power_of_two_the_flat_space_holds() -> None:
    """A 96-thread tile axis leaves a 16-element reduction 10 threads: its
    chunk is 8, a power of two, where a chunk of 10 failed strict
    normalize."""
    spec = _cute_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=128))
    for block_id, size in ((0, 128), (1, 16)):
        spec.num_threads.append(NumThreadsSpec(block_id=block_id, size_hint=size))
    spec.reduction_loops.append(ReductionLoopSpec(block_id=1, size_hint=16))
    spec.reduction_block_ids.add(1)
    config = helion.Config(
        block_sizes=[128], num_threads=[96, 0], reduction_loops=[None]
    )
    spec.normalize(config)
    assert config.config["reduction_loops"] == [8]
    spec.normalize(helion.Config(**config.config))


def test_repair_refuses_a_tile_whose_bounded_minimum_keeps_it_wide() -> None:
    """With the inner block's minimum at 32 no halving of the bounded pair
    keeps it there, so the [32, 32] tile cannot shrink and the generated
    config is refused like an explicit one."""
    spec = _nested_reduction_spec(bounded=True)
    spec.block_sizes.block_id_lookup(1).update_min(32)
    with pytest.raises(InvalidConfig, match="no thread budget left"):
        _normalized(spec, block_sizes=[32, 32], reduction_loops=[None])


def _sum_rows_then_double(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        sums[tile] = torch.sum(x[tile, :], dim=-1)
    for tile_a, tile_b in hl.tile(y.size()):
        doubled[tile_a, tile_b] = y[tile_a, tile_b] * 2
    return sums, doubled


def _sum_rows_twice(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x_sums = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    y_sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile in hl.tile(x.size(0)):
        x_sums[tile] = torch.sum(x[tile, :], dim=-1)
    for tile in hl.tile(y.size(0)):
        y_sums[tile] = torch.sum(y[tile, :], dim=-1)
    return x_sums, y_sums


def test_tiles_of_several_root_loops_are_repaired_against_their_product() -> None:
    """Root loops run one at a time, and the codegen gives a reduction what
    the launch they share leaves it; the repair shrinks a rejected config
    against the product of the roots' tiles, a conservative bound on that
    launch: the config renders instead of being refused (a multi-root kernel
    keeps a valid default)."""
    cases = (
        # A 128-element row sum beside an elementwise [256, 256] tile in the
        # other root.
        (
            _sum_rows_then_double,
            (256, 256),
            [[32, 32, 32], [16, 16, 16], [8, 16, 16], [4, 32, 32]],
        ),
        # Two 128-element row sums: the default [32, 32].
        (_sum_rows_twice, (64, 128), [[32, 32]]),
    )
    with mock_cuda_unavailable():
        for fn, y_shape, candidates in cases:
            kernel = helion.kernel(
                fn, backend="cute", static_shapes=True, autotune_effort="none"
            )
            args = (torch.empty(64, 128), torch.empty(y_shape))
            bound = cpu_bind(kernel, args)
            for block_sizes in candidates:
                config = helion.Config(block_sizes=block_sizes)
                bound.config_spec.normalize(config, _fix_invalid=True)
                assert config.config["block_sizes"] != block_sizes
                assert "_launcher(" in bound.to_code(config)


def test_default_config_of_middle_dim_reduction_kernel_renders() -> None:
    def partials(x: torch.Tensor) -> torch.Tensor:
        splits, rows, channels = x.shape
        out = torch.empty([splits, channels], dtype=torch.float32, device=x.device)
        for tile_s, tile_c in hl.tile([splits, channels]):
            out[tile_s, tile_c] = torch.sum(x[tile_s, :, tile_c].to(torch.float32), 1)
        return out

    kernel = helion.kernel(
        partials, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(64, 128, 384, dtype=torch.bfloat16),))
        config = bound.config_spec.default_config()
        assert _tile_threads(bound.config_spec, config) <= 1024 // 32
        assert "_helion_partials" in bound.to_code(config)


def _online_softmax_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    row_sums = torch.empty([m], dtype=torch.float32, device=x.device)
    prob_sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            vals = x[tile_m, tile_n].to(torch.float32)
            new_max = torch.maximum(row_max, torch.amax(vals, dim=-1))
            row_sum = row_sum * torch.exp(row_max - new_max) + torch.sum(
                torch.exp(vals - new_max[:, None]), dim=-1
            )
            row_max = new_max
        row_sums[tile_m] = row_sum
        probs = torch.exp(x[tile_m, :].to(torch.float32) - row_max[:, None])
        prob_sums[tile_m] = torch.sum(probs / row_sum[:, None], dim=-1)
    return row_sums, prob_sums


def _center_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    return out


def _shift_narrow_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty_like(y)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_k in hl.tile(y.size(1)):
            out[tile_m, tile_k] = (y[tile_m, tile_k] + mean[:, None]).to(y.dtype)
    return out


def _center_and_scale_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    centered = torch.empty_like(x)
    scaled = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n):
            centered[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
        for tile_k in hl.tile(n):
            scaled[tile_m, tile_k] = (x[tile_m, tile_k] * mean[:, None]).to(x.dtype)
    return centered, scaled


def _two_nested_tile_sums(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        out[tile_m] = acc - mean
    return out


def _three_nested_tile_sums(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                for tile_k in hl.tile(2048):
                    c = torch.sum(x[tile_m, tile_k].to(torch.float32), dim=-1)
                    acc = acc + a * b * c
        out[tile_m] = acc - mean
    return out


def _center_rows_beside_another_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return out, sums


def _center_rows_in_a_fixed_loop_beside_another_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    out2 = torch.empty_like(y)
    for tile_a in hl.tile(y.size(0)):
        out2[tile_a, :] = y[tile_a, :] * torch.sum(y[tile_a, :], -1, keepdim=True)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_i in hl.tile(n):
            for _tile_j in hl.tile(64, block_size=2):
                out[tile_m, tile_i] = (x[tile_m, tile_i] - mean[:, None]).to(x.dtype)
    return out, out2


def _center_rows_beside_a_bounded_loop(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    firsts = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_b in hl.tile(tile_m.begin, tile_m.end):
            firsts[tile_b] = x[tile_b, 0]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    return out, firsts


def _center_rows_of_two_widths(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out2 = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    for tile_r in hl.tile(y.size(0)):
        mean2 = torch.mean(y[tile_r, :].to(torch.float32), dim=-1)
        for tile_c in hl.tile(y.size(1)):
            out2[tile_r, tile_c] = (y[tile_r, tile_c] - mean2[:, None]).to(y.dtype)
    return out, out2


def _center_rows_of_two_widths_sharing_blocks(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = z.shape
    out = torch.empty_like(z)
    out2 = torch.empty_like(z)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = (z[tile_m, tile_n] - mean[:, None]).to(z.dtype)
    for tile_m in hl.tile(m, block_size=block_m):
        mean2 = torch.mean(y[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n, block_size=block_n):
            out2[tile_m, tile_n] = (z[tile_m, tile_n] - mean2[:, None]).to(z.dtype)
    return out, out2


def _center_rows_from_the_last(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for i in hl.grid(m - 1, -1, -1):
        mean = torch.mean(x[i, :].to(torch.float32))
        for tile_n in hl.tile(n):
            out[i, tile_n] = (x[i, tile_n] - mean).to(x.dtype)
    return out


def _scale_rows_then_sum(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    hl.barrier()
    for tile_r in hl.tile(y.size(0)):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return out, sums


def _scale_rows_then_sum_sharing_rows(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    sums = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    block_m = hl.register_block_size(m)
    for tile_m in hl.tile(m, block_size=block_m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    hl.barrier()
    for tile_r in hl.tile(y.size(0), block_size=block_m):
        sums[tile_r] = torch.sum(y[tile_r, :].to(torch.float32), dim=-1)
    return out, sums


def _tile_sums_nested_both_ways(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_i = hl.register_block_size(2048)
    block_j = hl.register_block_size(2048)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048, block_size=block_i):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048, block_size=block_j):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        for tile_j in hl.tile(2048, block_size=block_j):
            b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
            for tile_i in hl.tile(2048, block_size=block_i):
                a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
                acc = acc + a * b
        out[tile_m] = acc - mean
    return out


def _tile_sums_nested_and_alone(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_j = hl.register_block_size(2048)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048, block_size=block_j):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        for tile_j in hl.tile(2048, block_size=block_j):
            acc = acc + torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
        out[tile_m] = acc - mean
    return out


def _tile_sums_nested_and_alone_two_wide(x: torch.Tensor) -> torch.Tensor:
    """``_tile_sums_nested_and_alone`` with the shared block at least 2."""
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_j = hl.register_block_size(2, 2048)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048, block_size=block_j):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        for tile_j in hl.tile(2048, block_size=block_j):
            acc = acc + torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
        out[tile_m] = acc - mean
    return out


def _center_tunable_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    rows = hl.register_tunable("rows", PowerOfTwoFragment(32, 32))
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=helion.next_power_of_2(rows)):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    return out


def _tile_sums_nested_in_themselves(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block = hl.register_block_size(2048)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048, block_size=block):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048, block_size=block):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        out[tile_m] = acc - mean
    return out


def _two_nested_tile_sums_over_fixed_rows(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=32):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        out[tile_m] = acc - mean
    return out


def _tile_sums_in_a_fixed_loop(x: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_i in hl.tile(2048, block_size=64):
            a = torch.sum(x[tile_m, tile_i].to(torch.float32), dim=-1)
            for tile_j in hl.tile(2048):
                b = torch.sum(x[tile_m, tile_j].to(torch.float32), dim=-1)
                acc = acc + a * b
        out[tile_m] = acc - mean
    return out


def _center_scale_and_shift_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    centered = torch.empty_like(x)
    scaled = torch.empty_like(x)
    shifted = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n):
            centered[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
        for tile_k in hl.tile(n):
            scaled[tile_m, tile_k] = (x[tile_m, tile_k] * mean[:, None]).to(x.dtype)
        for tile_j in hl.tile(n):
            shifted[tile_m, tile_j] = (x[tile_m, tile_j] + mean[:, None]).to(x.dtype)
    return centered, scaled, shifted


def _center_and_scale_rows_sharing_a_block(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    centered = torch.empty_like(x)
    scaled = torch.empty_like(x)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n, block_size=block_n):
            centered[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            scaled[tile_m, tile_n] = (x[tile_m, tile_n] * mean[:, None]).to(x.dtype)
    return centered, scaled


def _center_rows_then_double_sharing_blocks(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    centered = torch.empty_like(x)
    doubled = torch.empty_like(y)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_n in hl.tile(n, block_size=block_n):
            centered[tile_m, tile_n] = (x[tile_m, tile_n] - mean[:, None]).to(x.dtype)
    for tile_m in hl.tile(m, block_size=block_m):
        for tile_n in hl.tile(n, block_size=block_n):
            doubled[tile_m, tile_n] = y[tile_m, tile_n] * 2
    return centered, doubled


def _shift_square_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = x.size(0)
    out = torch.empty_like(y)
    for tile_m in hl.tile(m):
        mean = torch.mean(x[tile_m, :].to(torch.float32), dim=-1)
        for tile_a, tile_b in hl.tile([y.size(1), y.size(2)]):
            out[tile_m, tile_a, tile_b] = (
                y[tile_m, tile_a, tile_b] + mean[:, None, None]
            ).to(y.dtype)
    return out


@pytest.mark.parametrize(
    ("fn", "shapes", "block_sizes", "launch"),
    [
        # The loop walks the reduced row: it and the reduction get a warp each.
        (_online_softmax_stats, [(64, 8192)], [1, 32], "(32, 32, 1)"),
        (_center_rows, [(64, 8192)], [1, 32], "(32, 32, 1)"),
        # Nested loops each take an axis: each is sized against the threads
        # its enclosing loops leave.  Sizing each of three against the full
        # budget would give 16 x 16 x 16 threads, which the render rejects.
        # Then the outer ones give up threads until the reduction keeps 16:
        # the shares alone, 16 x 4 x 2, would leave it 8.
        (_two_nested_tile_sums, [(64, 8192)], [1, 16, 4], "(16, 16, 4)"),
        (_three_nested_tile_sums, [(64, 8192)], [1, 8, 4, 2], "(16, 8, 4)"),
        # A fixed block size counts at its size.  Rows of 32 leave the
        # reduction 16 threads and the loops 1 x 2 (16 x 4 loop threads would
        # make a 2048-thread block, which the render rejects), and a loop of
        # 64 leaves 16 to the reduction and one to the loop inside it (its 16
        # would leave the reduction one).
        (_two_nested_tile_sums_over_fixed_rows, [(64, 8192)], [1, 2], "(16, 32, 2)"),
        (_tile_sums_in_a_fixed_loop, [(64, 8192)], [1, 1], "(16, 64, 1)"),
        # A fixed size of tunables has no thread count before a config: the
        # loops keep the collapsed default, and the reduction the warp it
        # would lose to a loop share were those rows counted as one thread.
        (_center_tunable_rows, [(64, 8192)], [1], "(32, 32, 1)"),
        # Sibling loops multiply in the reduction's thread count, although
        # they share one thread axis, and each runs as many threads as the
        # widest: two or three siblings keep 8 or 4 threads each (16 each would
        # leave the reduction 4 and one).  Loops sharing a block size, in one
        # root or in two, count once per loop (32 would leave it one).
        (_center_and_scale_rows, [(64, 8192)], [1, 8, 8], "(16, 8, 1)"),
        (_center_scale_and_shift_rows, [(64, 8192)], [1, 4, 4, 4], "(16, 4, 1)"),
        (_center_and_scale_rows_sharing_a_block, [(64, 8192)], [8, 1], "(16, 8, 1)"),
        (
            _center_rows_then_double_sharing_blocks,
            [(64, 8192), (64, 8192)],
            [1, 8],
            "(16, 8, 1)",
        ),
        # A reduction off a loop's paths, here the other root's over another
        # extent, takes threads of the shared launch beside the loop's own
        # reduction (``CuteBackend.adjust_reduction_thread_count``), which
        # the shares do not count: the loops keep the collapsed default.
        (
            _center_rows_of_two_widths,
            [(64, 8192), (64, 16384)],
            [1, 1, 1, 1],
            "(1024, 1, 1)",
        ),
        # The same where the roots share their blocks: both reductions sit
        # beside each root's loop.  Modelled as one 16 x 8192-wide reduction,
        # the share `[1, 8]` left the 8192-wide mean one thread.
        (
            _center_rows_of_two_widths_sharing_blocks,
            [(64, 16), (64, 8192), (64, 8192)],
            [1, 1],
            "(16, 64, 1)",
        ),
        # Counted, the first root's 16-thread sum left the second root's
        # reduction 2 threads against a 16-wide axis, which the render
        # rejects.
        (
            _center_rows_in_a_fixed_loop_beside_another_sum,
            [(64, 8192), (64, 16)],
            [1, 1, 1],
            "(16, 1, 1)",
        ),
        # A two-dimensional loop is balanced as a whole, 8 x 4 threads beside
        # the reduction's warp (sized block by block it would get 4 x 2, 2048
        # serial steps against the reduction's 64).
        (_shift_square_tiles, [(64, 8192), (64, 128, 128)], [1, 8, 4], "(32, 8, 4)"),
        # A loop the reduction dwarfs stays serial beside its 1024 threads.
        (_shift_narrow_rows, [(64, 65536), (64, 64)], [1, 1], "(1024, 1, 1)"),
        # A row of at most 256 elements never collapsed the defaults to 1.
        (_center_rows, [(64, 256)], [16, 16], "(4, 16, 16)"),
        # The same for another root's reduction over a wider extent: a share
        # would also take its threads, (64, 1, 16) for its 65536-wide sum.
        (
            _center_rows_beside_another_sum,
            [(64, 8192), (64, 65536)],
            [1, 1, 1],
            "(1024, 1, 1)",
        ),
        # A loop bounded by the row tile spans one row and gets no share: the
        # sibling loop keeps its own (16 of the 1024 threads where a kernel of
        # three blocks defaults each to 16).
        (_center_rows_beside_a_bounded_loop, [(64, 8192)], [1, 1, 16], "(64, 16, 1)"),
        # A loop with no reduction in its own root keeps the collapsed default,
        # also when the roots share their row block: the second root's sum
        # still runs on its own grid.
        (_scale_rows_then_sum, [(64, 8192), (64, 8192)], [1, 1, 1], "(1024, 1, 1)"),
        (
            _scale_rows_then_sum_sharing_rows,
            [(64, 8192), (64, 65536)],
            [1, 1],
            "(1024, 1, 1)",
        ),
        # Loops sharing block sizes in both nesting orders, or one shared block
        # nested and alone, would need a block on two thread axes: those
        # blocks keep the collapsed default and the others get their share.
        (_tile_sums_nested_both_ways, [(64, 8192)], [1, 1, 1], "(1024, 1, 1)"),
        (_tile_sums_nested_and_alone, [(64, 8192)], [1, 1, 16], "(64, 16, 1)"),
        # A shared block of at least 2 threads still takes an axis, after the
        # loop around it on one path only: that loop gets no share either.
        (
            _tile_sums_nested_and_alone_two_wide,
            [(64, 8192)],
            [2, 1, 1],
            "(256, 2, 1)",
        ),
    ],
)
def test_default_config_balances_inner_tile_loop_and_wide_reduction(
    fn: Callable[..., object],
    shapes: list[tuple[int, ...]],
    block_sizes: list[int],
    launch: str,
) -> None:
    """Where a wide full-slice reduction collapses the default block sizes to
    1 (one row per CTA), an inner ``hl.tile`` loop takes the thread share that
    balances its serial steps against the reductions on its own path, and the
    reduction keeps at least 16 threads.  Block size 1 walked the online
    softmax over 262144 columns one column per step in every thread, and its
    262144-term running fp32 sum missed the fp64-anchored bound; a reduction
    left one thread sums every column of its row alone.  A reduction in
    another root (here the second root, after an ``hl.barrier()`` in two
    cases) runs on its own grid and is not counted, but one over another
    extent shares the launch with the loop's own and keeps the defaults
    collapsed.  The autotuner's reference config (its accuracy baseline) is
    the same config."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    args = tuple(torch.randn(*shape, dtype=torch.bfloat16) for shape in shapes)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        config = spec.default_config()
        assert config.block_sizes == block_sizes
        assert spec.autotune_reference_config().block_sizes == block_sizes
        assert f"block={launch}" in bound.to_code(config)


def test_a_block_nested_in_itself_keeps_the_collapsed_default() -> None:
    """One registered block on two nested loops of a path would need two
    thread axes: it takes no share, so the default stays collapsed and the
    compiler reports the nested loops' conflict, not a thread-axis
    mismatch."""
    kernel = helion.kernel(
        _tile_sums_nested_in_themselves, backend="cute", static_shapes=True
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (torch.randn(64, 8192, dtype=torch.bfloat16),))
        config = bound.config_spec.default_config()
        assert config.block_sizes == [1, 1]
        with pytest.raises(NestedDeviceLoopsConflict):
            bound.to_code(config)


@skipIfRefEager("binding does not build device IR in ref eager mode")
def test_a_descending_grid_keeps_the_collapsed_default() -> None:
    """A descending ``hl.grid``'s step of -1 counts no threads, like a fixed
    size of tunables: the loop keeps the collapsed default.  Counted, it
    failed the bind in ``math.isqrt``, an explicit config's too."""
    kernel = helion.kernel(
        _center_rows_from_the_last, backend="cute", static_shapes=True
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, (torch.randn(64, 8192, dtype=torch.bfloat16),))
        assert bound.config_spec.default_config().block_sizes == [1]
        code = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[32], num_threads=[1, 32], reduction_loops=[4096]
                )
            )
        )
    assert "block=(32, 1, 1)" in code


@pytest.mark.cuda
@skipIfNotCUDA()
def test_default_config_shares_match_eager() -> None:
    """The defaults that give an inner loop a share of the threads beside a
    wide reduction (``[1, 32]`` at 64x8192) compute what eager PyTorch does:
    the online softmax statistics and the centered rows."""
    x = torch.randn(64, 8192, device=DEVICE, dtype=torch.bfloat16)
    stats = helion.kernel(
        _online_softmax_stats,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    centered = helion.kernel(
        _center_rows, backend="cute", static_shapes=True, autotune_effort="none"
    )
    for kernel in (stats, centered):
        assert kernel.bind((x,)).config_spec.default_config().block_sizes == [1, 32]
    row_sums, prob_sums = stats(x)
    data = x.float()
    torch.testing.assert_close(
        row_sums,
        torch.exp(data - data.amax(-1, keepdim=True)).sum(-1),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        prob_sums, torch.ones_like(prob_sums), rtol=1e-5, atol=1e-5
    )
    # Within a bf16 rounding of the fp32 difference.
    torch.testing.assert_close(
        centered(x).float(),
        (data - data.mean(-1, keepdim=True)).to(torch.bfloat16).float(),
        rtol=1e-2,
        atol=1e-2,
    )
