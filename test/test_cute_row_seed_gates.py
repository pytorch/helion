"""The gate of the CuTe row heuristics (``_row_tile_groups``): they seed
layouts of the row of the kernel's reduction loop, so they decline a kernel
that reduces a second block, whose row layouts the lowering refuses, and a
kernel with a block size that is no tile axis of the rows."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune.heuristics import cute as cute_heuristics
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteMaskedRolledClusterHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute import CuteReductionTileHeuristic
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteReductionWideChunkHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteRolledClusterLadderHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute import CuteRolledRowLadderHeuristic
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteRolledRowLoadsFirstHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute import _is_sole_row_reduction
from helion._compiler.cute.autotune.heuristics.cute import _row_tile_axes
from helion._compiler.cute.autotune.heuristics.cute import _row_tile_groups
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_row_seed_kernels import first_elements_then_row_norm
from helion._testing.cute_row_seed_kernels import grid_first_elements_then_row_norm
from helion._testing.cute_row_seed_kernels import grid_row_norm_then_tiled_columns
from helion._testing.cute_row_seed_kernels import half_then_row_norms
from helion._testing.cute_row_seed_kernels import row_and_half_norms
from helion._testing.cute_row_seed_kernels import row_and_half_norms_over_two_tiles
from helion._testing.cute_row_seed_kernels import (
    row_norm_beside_a_loop_sharing_a_block_size,
)
from helion._testing.cute_row_seed_kernels import row_norm_beside_doubled_rows
from helion._testing.cute_row_seed_kernels import (
    row_norm_over_two_tiles_then_doubled_rows,
)
from helion._testing.cute_row_seed_kernels import row_norm_then_grid_doubled_rows
from helion._testing.cute_row_seed_kernels import row_norm_then_grid_row_norm
from helion._testing.cute_row_seed_kernels import row_norm_then_tiled_columns
from helion._testing.cute_row_seed_kernels import row_norm_then_tiled_doubled_rows
from helion._testing.cute_row_seed_kernels import rows_over_two_sums
from helion._testing.cute_row_seed_kernels import softmax_rows
from helion._testing.cute_row_seed_kernels import softmax_rows_inner_block_size_first
from helion._testing.cute_row_seed_kernels import softmax_rows_over_two_tiles
from helion._testing.cute_row_seed_kernels import (
    tiled_first_elements_then_grid_doubled_rows,
)
from helion._testing.cute_row_seed_kernels import (
    tiled_first_elements_then_grid_row_norm,
)
from helion._testing.cute_row_seed_kernels import two_row_norms_in_swapped_tile_orders
from helion._testing.cute_row_seed_kernels import two_row_norms_over_tiles_of_their_own
from helion._testing.cute_row_seed_kernels import two_row_norms_sharing_a_block_size
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion._compiler.autotuner_heuristics.registry import AutotunerHeuristic
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

# The heuristics behind the gate, in registry order.  The first five decline
# a kernel it declines; the loads-first rows keep their knob there.
ROW_HEURISTICS: tuple[type[AutotunerHeuristic], ...] = (
    CuteReductionTileHeuristic,
    CuteReductionWideChunkHeuristic,
    CuteRolledRowLadderHeuristic,
    CuteRolledClusterLadderHeuristic,
    CuteMaskedRolledClusterHeuristic,
    CuteRolledRowLoadsFirstHeuristic,
)
LAYOUT_HEURISTICS = ROW_HEURISTICS[:-1]
# What the loads-first rows seed where the gate declines their layout.
UNROLL_KNOB = Config(cute_lane_unroll=16)


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_target():
        yield


def _bind(fn: Callable[..., object], *shapes: tuple[int, ...]) -> BoundKernel:
    """``fn`` bound on CPU to bf16 tensors of ``shapes``; a one-dim argument
    is float32."""
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        ignore_warnings=[helion.exc.FastMathUnused],
    )
    args = tuple(
        torch.empty(*shape, dtype=torch.float32 if len(shape) == 1 else torch.bfloat16)
        for shape in shapes
    )
    with cpu_target():
        return cpu_bind(kernel, args)


@functools.cache
def _bound(fn: Callable[..., object], *shapes: tuple[int, ...]) -> BoundKernel:
    """``_bind``, once a module."""
    return _bind(fn, *shapes)


def _seeds(heuristic: type[AutotunerHeuristic], bound: BoundKernel) -> list[Config]:
    """The seeds the registry takes from ``heuristic`` for ``bound``."""
    env, device_ir = bound.env, bound.host_function.device_ir
    if not heuristic.is_eligible(env, device_ir):
        return []
    seeds = heuristic.get_seed_configs(env, device_ir)
    if seeds is None:
        seeds = [heuristic.get_seed_config(env, device_ir)]
    return [seed for seed in seeds if seed is not None]


def _refusal(bound: BoundKernel, seed: Config) -> str | None:
    """Why the search cannot compile ``seed``, ``None`` when it renders: the
    seed does not transfer to the search space (a list sized for a kernel
    with one reduced block), or the lowering refuses its layout."""
    generation = bound.config_spec.create_config_generation()
    try:
        _, normalized = generation.canonicalize_flat(generation.flatten(seed))
    except AssertionError:
        return "no transfer"
    try:
        bound.to_code(normalized)
    except exc.BackendUnsupported as error:
        return str(error).splitlines()[0]
    return None


def _fired(bound: BoundKernel) -> list[str]:
    """The row heuristics among the heuristics that seeded ``bound``."""
    names = {heuristic.name for heuristic in ROW_HEURISTICS}
    return [name for name in bound.config_spec.autotuner_heuristics if name in names]


def _block_sizes_in_order(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[list[int]]:
    """What the heuristics took for the rows' tile axes before the gate:
    every block size, in the order registered."""
    return [list(env.config_spec.block_sizes.valid_block_ids())]


def _without_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cute_heuristics, "_row_tile_groups", _block_sizes_in_order)


def _ungated(
    monkeypatch: pytest.MonkeyPatch, bound: BoundKernel
) -> dict[str, list[Config]]:
    """The seeds each row heuristic emits for ``bound`` without the gate."""
    with monkeypatch.context() as patched:
        _without_the_gate(patched)
        emitted = {
            heuristic.name: _seeds(heuristic, bound) for heuristic in ROW_HEURISTICS
        }
    return {name: seeds for name, seeds in emitted.items() if seeds}


# ``groups`` are the tile axes of the root loops over the rows.
@pytest.mark.parametrize(
    ("kernel", "shapes", "groups", "fired", "warp_rows"),
    [
        pytest.param(
            softmax_rows,
            [(2048, 256)],
            [[0]],
            [
                "cute_reduction_tile",
                "cute_rolled_row_ladder",
                "cute_rolled_row_loads_first",
            ],
            [[2], [4]],
            id="one tile axis",
        ),
        pytest.param(
            softmax_rows_over_two_tiles,
            [(8, 256, 256)],
            [[0, 1]],
            ["cute_reduction_tile"],
            [[1, 2], [1, 4]],
            id="two tile axes",
        ),
        pytest.param(
            # The hl.grid loop has no block size: the kernel's one block
            # size tiles the reduced rows.
            grid_first_elements_then_row_norm,
            [(2048, 256), (2048,)],
            [[1]],
            [
                "cute_reduction_tile",
                "cute_rolled_row_ladder",
                "cute_rolled_row_loads_first",
            ],
            [[2], [4]],
            id="a root loop without a block size beside the row loop",
        ),
        pytest.param(
            # The hl.grid loop sweeps rows of the same width, so it shares
            # the row's block; it has no block size.
            row_norm_then_grid_doubled_rows,
            [(2048, 256)],
            [[0]],
            [
                "cute_reduction_tile",
                "cute_rolled_row_ladder",
                "cute_rolled_row_loads_first",
            ],
            [[2], [4]],
            id="a root loop without a block size that sweeps the rows too",
        ),
        pytest.param(
            # The hl.grid loop reduces the rows as well, one row a CTA.
            row_norm_then_grid_row_norm,
            [(2048, 256)],
            [[0]],
            [
                "cute_reduction_tile",
                "cute_rolled_row_ladder",
                "cute_rolled_row_loads_first",
            ],
            [[2], [4]],
            id="a root loop without a block size that reduces the rows too",
        ),
        pytest.param(
            two_row_norms_sharing_a_block_size,
            [(2048, 256)],
            [[0]],
            [
                "cute_reduction_tile",
                "cute_rolled_row_ladder",
                "cute_rolled_row_loads_first",
            ],
            [[2], [4]],
            id="two row loops with one block size",
        ),
        pytest.param(
            # Each loop tiles its rows with a block size of its own; one row
            # per CTA is one row of either.  The warp rows have no one axis
            # for their rows per CTA.
            two_row_norms_over_tiles_of_their_own,
            [(2048, 256), (512, 256)],
            [[0], [2]],
            ["cute_reduction_tile"],
            [],
            id="two row loops with a block size each",
        ),
        pytest.param(
            # The second loop sweeps the rows without reducing them, over a
            # block size of its own.
            row_norm_then_tiled_doubled_rows,
            [(2048, 256)],
            [[0], [2]],
            ["cute_reduction_tile"],
            [],
            id="a row loop and a loop that sweeps the rows, a block size each",
        ),
        pytest.param(
            softmax_rows,
            [(4, 65536)],
            [[0]],
            [
                "cute_reduction_tile",
                "cute_reduction_wide_chunk",
                "cute_rolled_row_ladder",
                "cute_rolled_cluster_ladder",
                "cute_masked_rolled_cluster",
                "cute_rolled_row_loads_first",
            ],
            [],
            id="wide rows",
        ),
    ],
)
def test_sole_row_kernels_keep_their_seeds(
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
    groups: list[list[int]],
    fired: list[str],
    warp_rows: list[list[int]],
) -> None:
    bound = _bound(kernel, *shapes)
    env, device_ir = bound.env, bound.host_function.device_ir
    assert _row_tile_groups(env, device_ir) == groups
    assert _row_tile_axes(env, device_ir) == (groups[0] if len(groups) == 1 else None)
    assert _is_sole_row_reduction(env, device_ir)
    assert _fired(bound) == fired
    assert [
        seed.config["block_sizes"]
        for seed in CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir)
    ] == warp_rows
    # The first seed of each heuristic, and the reduction tile's last (a
    # warp row where it has them), render.
    for heuristic in ROW_HEURISTICS:
        seeds = _seeds(heuristic, bound)
        assert bool(seeds) == (heuristic.name in fired)
        assert UNROLL_KNOB not in seeds
        for seed in seeds[:1] + seeds[1:][-1:]:
            assert seed in bound.config_spec.compiler_seed_configs
            assert _refusal(bound, seed) is None


def test_warp_rows_take_the_innermost_tile_axis() -> None:
    # The inner tile axis, block 0, takes a block size registered before the
    # loop, so the rows' tile axes are blocks 1 then 0 and the last block
    # size is the outer one.
    bound = _bound(softmax_rows_inner_block_size_first, (8, 256, 256))
    spec = bound.config_spec
    env, device_ir = bound.env, bound.host_function.device_ir
    assert [item.block_id for item in spec.block_sizes] == [0, 1]
    assert device_ir.grid_block_ids == [[1, 0]]
    assert _row_tile_axes(env, device_ir) == [1, 0]
    seeds = CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir)
    assert [seed.config["block_sizes"] for seed in seeds] == [[2, 1], [4, 1]]
    for seed, rows in zip(seeds, (2, 4), strict=True):
        threads = seed.config["num_threads"]
        assert spec.num_threads.config_get(threads, 0) == rows
        assert spec.num_threads.config_get(threads, 1) == 0
        assert seed in spec.compiler_seed_configs
    code = bound.to_code(seeds[0])
    assert "block=(32, 2, 1)" in code


# Without the gate each row heuristic emits ``emitted`` seeds for the kernel,
# ``refused`` of which the search cannot compile.  The counts are the gate's
# reason, pinned as an alarm: when the lowering learns one of these layouts
# beside a second reduced block, its count drops here and the gate, which
# would then withhold a seed that compiles, is due for another look.
# ``knob`` says whether the loads-first rows, which need one tile axis and a
# row store, keep their knob.
@pytest.mark.parametrize(
    ("kernel", "shapes", "row", "reduced", "knob", "ungated"),
    [
        pytest.param(
            row_and_half_norms,
            [(2048, 256)],
            1,
            {1, 2},
            True,
            {
                "cute_reduction_tile": (4, 4),
                "cute_rolled_row_ladder": (1, 1),
                "cute_rolled_row_loads_first": (1, 1),
            },
            id="half row beside the row",
        ),
        pytest.param(
            half_then_row_norms,
            [(2048, 256)],
            1,
            {1, 2},
            True,
            {
                "cute_reduction_tile": (1, 1),
                "cute_rolled_row_ladder": (1, 1),
                "cute_rolled_row_loads_first": (1, 1),
            },
            id="half row reduced first",
        ),
        pytest.param(
            rows_over_two_sums,
            [(2048, 256), (2048, 64)],
            1,
            {1, 2},
            True,
            {
                "cute_reduction_tile": (4, 4),
                "cute_rolled_row_ladder": (1, 1),
                "cute_rolled_row_loads_first": (1, 1),
            },
            id="rows of a second tensor",
        ),
        pytest.param(
            row_and_half_norms_over_two_tiles,
            [(64, 8, 256)],
            2,
            {2, 3},
            False,
            {"cute_reduction_tile": (4, 4)},
            id="half row beside the row, two tile axes",
        ),
        pytest.param(
            # The narrow chunk and the wide chunk, which leave every thread
            # count to the backend, render at this width.
            row_and_half_norms,
            [(4, 65536)],
            1,
            {1, 2},
            True,
            {
                "cute_reduction_tile": (2, 1),
                "cute_reduction_wide_chunk": (1, 0),
                "cute_rolled_row_ladder": (1, 1),
                "cute_rolled_cluster_ladder": (1, 1),
                "cute_masked_rolled_cluster": (12, 12),
                "cute_rolled_row_loads_first": (1, 1),
            },
            id="half row beside a wide row",
        ),
    ],
)
def test_second_reduced_block_gets_no_row_layout(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
    row: int,
    reduced: set[int],
    knob: bool,
    ungated: dict[str, tuple[int, int]],
) -> None:
    bound = _bound(kernel, *shapes)
    spec = bound.config_spec
    env, device_ir = bound.env, bound.host_function.device_ir
    fact = spec.reduction_kernel_fact
    assert fact is not None
    # One reduction loop, on ``row``; the other reduced block has none.
    assert [loop.block_id for loop in spec.reduction_loops] == [row]
    assert {descriptor.block_id for descriptor in fact.reductions} == reduced
    assert _row_tile_groups(env, device_ir) is None
    assert not _is_sole_row_reduction(env, device_ir)
    for heuristic in LAYOUT_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir) == []
    kept = [UNROLL_KNOB] if knob else []
    assert _seeds(CuteRolledRowLoadsFirstHeuristic, bound) == kept
    assert _fired(bound) == [CuteRolledRowLoadsFirstHeuristic.name] * knob
    emitted = _ungated(monkeypatch, bound)
    assert {
        name: (
            len(seeds),
            sum(_refusal(bound, seed) is not None for seed in seeds),
        )
        for name, seeds in emitted.items()
    } == ungated


def test_kernel_with_a_second_reduced_block_keeps_seeds_that_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = _bound(row_and_half_norms, (2048, 256))
    seeds = _ungated(monkeypatch, bound)["cute_reduction_tile"]
    # Without the gate: the narrow chunk, the one-vector row, and the warp
    # rows at 2 and 4 rows per CTA, none of which the lowering takes.
    assert [seed.config["block_sizes"] for seed in seeds] == [[1], [1], [2], [4]]
    assert all(_refusal(bound, seed) is not None for seed in seeds)
    # The seeds the kernel keeps render: the loads-first knob, and the seeds
    # of a heuristic that lays out no row.
    kept = bound.config_spec.compiler_seed_configs
    assert kept[0] == UNROLL_KNOB
    assert len(kept) > 1
    assert [_refusal(bound, seed) for seed in kept] == [None] * len(kept)


def test_masked_cluster_seeds_fill_every_reload_slot() -> None:
    # The 48-wide rows of ``y``, swept beside the reduced row, have a reload
    # slot of their own, so the kernel's reload list has two entries.
    bound = _bound(row_norm_beside_doubled_rows, (64, 32768), (64, 48))
    spec = bound.config_spec
    assert _row_tile_groups(bound.env, bound.host_function.device_ir) == [[0]]
    (row,) = (loop.block_id for loop in spec.reduction_loops)
    assert len(spec.cute_reduction_reloads) == 2
    seeds = _seeds(CuteMaskedRolledClusterHeuristic, bound)
    assert seeds
    for seed in seeds:
        reloads = seed.config["cute_reduction_reloads"]
        assert len(reloads) == 2
        assert spec.cute_reduction_reloads.config_get(reloads, row) == "register"
    # The search takes them: a list of one entry did not transfer.
    for seed in (seeds[0], seeds[-1]):
        assert seed in spec.compiler_seed_configs
        assert _refusal(bound, seed) is None


# The gate drops seeds; the knob the loads-first rows open stays searchable,
# and with it the search space of a kernel of either kind.
@pytest.mark.parametrize(
    ("kernel", "shapes"),
    [
        pytest.param(row_and_half_norms, [(2048, 256)], id="second reduced block"),
        pytest.param(
            grid_row_norm_then_tiled_columns,
            [(2048, 256), (2048, 16)],
            id="block size off the row tiles",
        ),
    ],
)
def test_declined_kernels_keep_the_lane_unroll_knob(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
) -> None:
    bound = _bound(kernel, *shapes)
    spec = bound.config_spec
    assert not _is_sole_row_reduction(bound.env, bound.host_function.device_ir)
    assert _seeds(CuteRolledRowLoadsFirstHeuristic, bound) == [UNROLL_KNOB]
    assert UNROLL_KNOB in spec.compiler_seed_configs
    assert _refusal(bound, UNROLL_KNOB) is None
    unroll = spec._flat_fields()["cute_lane_unroll"]
    assert unroll.search_choices == (1, 2, 4, 8, 16)
    # Bound without the gate, the kernel has the same search space.
    _without_the_gate(monkeypatch)
    ungated = _bind(kernel, *shapes).config_spec
    assert CuteReductionTileHeuristic.name in ungated.autotuner_heuristics
    assert ungated.structural_fingerprint() == spec.structural_fingerprint()


# ``block_sizes`` are the block ids of the kernel's block sizes, ``extents``
# whether each tiles one extent, and ``grids`` the axes of its root loops.
# Without the gate the warp rows give ``rows_on`` their rows per CTA.
# ``knob`` as above.
@pytest.mark.parametrize(
    ("kernel", "shapes", "block_sizes", "extents", "grids", "row", "rows_on", "knob"),
    [
        pytest.param(
            row_norm_then_tiled_columns,
            [(2048, 256), (2048, 16)],
            [0, 2],
            [True, True],
            [[0]],
            1,
            2,
            False,
            id="tile loop nested in the row loop",
        ),
        pytest.param(
            grid_row_norm_then_tiled_columns,
            [(2048, 256), (2048, 16)],
            [2],
            [True],
            [[0]],
            1,
            2,
            True,
            id="tile loop nested in a grid loop over the rows",
        ),
        pytest.param(
            first_elements_then_row_norm,
            [(2048, 256), (2048,)],
            [0, 1],
            [True, True],
            [[0], [1]],
            2,
            1,
            False,
            id="tile of another root loop",
        ),
        pytest.param(
            tiled_first_elements_then_grid_row_norm,
            [(2048, 256), (2048,)],
            [0],
            [True],
            [[0], [1]],
            2,
            0,
            True,
            id="tile of the root loop without the reduction",
        ),
        pytest.param(
            # Both block sizes are axes of the row loop; block 0 also tiles
            # the 48 elements of ``y`` in a second root loop.
            row_norm_beside_a_loop_sharing_a_block_size,
            [(8, 256, 256), (48,)],
            [0, 1],
            [False, True],
            [[1, 0], [0]],
            2,
            1,
            False,
            id="block size shared with a loop of another extent",
        ),
    ],
)
def test_block_size_off_the_row_tiles_gets_no_row_layout(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
    block_sizes: list[int],
    extents: list[bool],
    grids: list[list[int]],
    row: int,
    rows_on: int,
    knob: bool,
) -> None:
    bound = _bound(kernel, *shapes)
    spec = bound.config_spec
    env, device_ir = bound.env, bound.host_function.device_ir
    fact = spec.reduction_kernel_fact
    assert fact is not None
    assert [item.block_id for item in spec.block_sizes] == block_sizes
    assert [
        isinstance(env.block_sizes[block_id].size, (int, torch.SymInt))
        for block_id in block_sizes
    ] == extents
    assert device_ir.grid_block_ids == grids
    # The row is the one block the kernel reduces: only the block sizes
    # decline it.
    assert [loop.block_id for loop in spec.reduction_loops] == [row]
    assert {descriptor.block_id for descriptor in fact.reductions} == {row}
    assert _row_tile_groups(env, device_ir) is None
    for heuristic in LAYOUT_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir) == []
    kept = [UNROLL_KNOB] if knob else []
    assert _seeds(CuteRolledRowLoadsFirstHeuristic, bound) == kept
    assert _fired(bound) == [CuteRolledRowLoadsFirstHeuristic.name] * knob
    # Without the gate the warp rows hold 2 and 4 "rows" of ``rows_on``,
    # with as many threads on it.
    warp_rows = _ungated(monkeypatch, bound)["cute_reduction_tile"][-2:]
    index = spec.block_sizes.block_id_to_index(rows_on)
    for seed, rows in zip(warp_rows, (2, 4), strict=True):
        assert seed.config["block_sizes"][index] == rows
        assert spec.num_threads.config_get(seed.config["num_threads"], rows_on) == rows


# Both block sizes tile one extent and are axes of a root loop that reduces
# the rows, the one block the kernel reduces: only the axes of the other root
# loop over rows of that width, ``grids[1]``, decline it.  They share a block
# size with the first loop's without being the same axes.
@pytest.mark.parametrize(
    ("kernel", "shapes", "grids"),
    [
        pytest.param(
            two_row_norms_in_swapped_tile_orders,
            [(8, 16, 256), (16, 8, 256)],
            [[0, 1], [1, 0]],
            id="row loops with the tile axes in two orders",
        ),
        pytest.param(
            row_norm_over_two_tiles_then_doubled_rows,
            [(8, 16, 256), (8, 256)],
            [[0, 1], [0]],
            id="a loop that sweeps the rows over one of their tile axes",
        ),
    ],
)
def test_row_loops_with_other_tile_axes_get_no_row_layout(
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
    grids: list[list[int]],
) -> None:
    bound = _bound(kernel, *shapes)
    spec = bound.config_spec
    env, device_ir = bound.env, bound.host_function.device_ir
    fact = spec.reduction_kernel_fact
    assert fact is not None
    assert [item.block_id for item in spec.block_sizes] == [0, 1]
    assert all(
        isinstance(env.block_sizes[block_id].size, (int, torch.SymInt))
        for block_id in (0, 1)
    )
    assert device_ir.grid_block_ids == grids
    assert [loop.block_id for loop in spec.reduction_loops] == [2]
    assert {descriptor.block_id for descriptor in fact.reductions} == {2}
    assert _row_tile_groups(env, device_ir) is None
    for heuristic in ROW_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir) == []
    assert _fired(bound) == []


def test_row_loop_without_a_reduction_is_found_by_its_rolled_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The grid loop rolls the row, block 2, which nothing reduces; the
    # kernel's one block size tiles the other root loop.
    bound = _bound(tiled_first_elements_then_grid_doubled_rows, (2048, 256), (2048,))
    spec = bound.config_spec
    env, device_ir = bound.env, bound.host_function.device_ir
    fact = spec.reduction_kernel_fact
    assert fact is None or not fact.reductions
    assert [item.block_id for item in spec.block_sizes] == [0]
    assert device_ir.grid_block_ids == [[0], [1]]
    assert [loop.block_id for loop in spec.reduction_loops] == [2]
    assert _row_tile_groups(env, device_ir) is None
    for heuristic in LAYOUT_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(env, device_ir) == []
    assert _seeds(CuteRolledRowLoadsFirstHeuristic, bound) == [UNROLL_KNOB]
    # Without the gate the warp rows hold 2 and 4 "rows" of that other loop.
    warp_rows = _ungated(monkeypatch, bound)["cute_reduction_tile"][-2:]
    assert [seed.config["block_sizes"] for seed in warp_rows] == [[2], [4]]
