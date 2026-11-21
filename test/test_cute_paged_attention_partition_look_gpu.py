"""The spare warp's look for a unit's lowest part changes no output bit
(``cute_mma_dag_partition_look="spare"``): alone and beside what its seeds
pair it with, eager and as captured calls that overlap their predecessors,
over units of unequal length that the balanced partition cuts into two
parts and into more, on grids of other sizes than the kernel was bound
for, and on every launch of a graph replayed over changing lengths, where
a counter or a staged pair left over from the launch ahead would show.

Each launch is also the test of the shared-memory accounting of the stage:
CUDA rejects a launch whose table the device does not hold."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._compiler.cute.paged_attention_shared_memory import ValueRingTiles
from helion._compiler.cute.paged_attention_shared_memory import partition_look_reason
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import launch_with_grid
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_stream import IDLE_SMS
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import check_same_bits
from helion._testing.cute_paged_stream import device_shared_memory
from helion._testing.cute_paged_stream import launched_bits
from helion._testing.cute_paged_stream import one_wave_units
from helion._testing.cute_paged_stream import poison_results
from helion._testing.cute_paged_stream import replayed_bits
from helion._testing.cute_paged_stream import stream_pair
from helion._testing.cute_paged_stream import stream_patterns
from helion._testing.poisoned_outputs import poisoned_launch
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = "cute_mma_dag_partition_look"
PARTITION = "cute_mma_dag_partition"
SKEW = "cute_mma_dag_partition_skew"
ORDER = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
V_LAG = "cute_mma_dag_v_lag"
AHEAD = "cute_mma_dag_boundary_prefetch_ahead"
TILES = "cute_mma_dag_value_ring_tiles"
BALANCED = {PARTITION: "balanced"}
SPARE = {KEY: "spare"}
WALK = {**BALANCED, ORDER: "reversed", LEAD: 8}
# CTAs a persistent launch of the one-wave fixture gets: its units (the
# device's SMs less ``IDLE_SMS``) outnumber them, so the grid holds more
# units than the kernel is bound for SMs and the balanced partition cuts
# units between CTAs: about two units a CTA on 64, where a unit of the
# full table lies in two CTAs and, between empty units, in three or four;
# five units to four CTAs on 100.
CTAS = 64
REPLAYS = 50


def _pair(
    candidate: dict[str, object],
    schedule: dict[str, object],
    *,
    window: int = PAGES * PAGE - 1,
    rows: int = 8,
    reference_scores: bool = False,
    ctas: int = CTAS,
):
    """The fixture's kernel on the balanced partition of ``ctas`` CTAs under
    ``schedule``, and the same kernel with ``candidate``'s options."""
    units = one_wave_units()
    if not ctas < units:
        pytest.skip("the grid needs more units than CTAs")
    columns = 8 if rows <= 8 else 32 if rows <= 32 else 40
    tiles = cast("int", {**schedule, **candidate}.get(TILES, 0))
    reason = partition_look_reason(
        device_shared_memory(),
        columns,
        query_stages=2,
        units=units,
        **({"value_ring": ValueRingTiles(tiles)} if tiles else {}),
    )
    if reason is not None:
        pytest.skip(reason)
    args, off, on, plain, source = stream_pair(
        candidate,
        schedule=schedule,
        window=window,
        rows=rows,
        reserved_sms=units + IDLE_SMS - ctas,
        reference_scores=reference_scores,
    )
    assert "partition_look" not in body_keywords(plain)
    keywords = body_keywords(source)
    assert keywords["partition_look"] == "'spare'"
    assert keywords["shared_memory_capacity"] == str(device_shared_memory())
    return args, off, on


def _check(
    candidate: dict[str, object],
    schedule: dict[str, object],
    *,
    calls: int = 3,
    **options: object,
) -> None:
    """Streams of every length the table has, eager and captured: the
    kernel whose spare warp looks stores the bits of the kernel whose
    compute group does."""
    args, off, on = _pair(candidate, schedule, **options)
    window = cast("int", options.get("window", PAGES * PAGE - 1))
    check_same_bits(
        args,
        off,
        on,
        stream_patterns(cast("torch.Tensor", args[6]).numel(), window),
        calls=calls,
        what=(candidate, schedule, options),
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_spare_warps_look_keeps_every_output_bit():
    """Units of zero to seventeen pages, about two to a CTA. A CTA's last
    range is a unit's lowest part in most CTAs, of a unit in two parts
    (the spare warp stages the other part's row statistics where it finds
    it complete) and, between units without pages, in three or four (the
    group folds every part from the workspace); its first range is a part
    it posts, a whole unit, or that lowest part itself, the one range of a
    CTA inside a long unit. Whichever part merges, the bits are those of
    the kernel whose group looks."""
    _check(SPARE, BALANCED)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_every_launch_of_a_replayed_graph_stores_the_same_bits():
    """Four captured calls back to back, each overlapping the one ahead,
    replayed fifty times while the requests' lengths change between
    replays: the units a CTA runs, the parts it posts and looks for and
    the units that are complete at the look differ from one replay to the
    next, and every launch of every replay stores the bits of an eager
    call of the kernel whose group looks. A look ahead of the launch's
    dependencies would read the counters of the launch ahead, and a
    counter or a role left over would merge a unit from the launch ahead's
    parts."""
    args, off, on = _pair(SPARE, BALANCED)
    units = cast("torch.Tensor", args[6]).numel()
    patterns = stream_patterns(units, PAGES * PAGE - 1)
    names = ("ragged", "even empty", "short", "full", "page edges")
    expected = {}
    for name in names:
        set_request_lengths(args, patterns[name])
        expected[name] = launched_bits(off, args)
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        captured = [cast("Callable[..., object]", on)(*args) for _ in range(4)]
    assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * 3, modes
    for replay in range(REPLAYS):
        name = names[replay % len(names)]
        set_request_lengths(args, patterns[name])
        poison_results(captured)
        graph.replay()
        torch.cuda.synchronize()
        for index, result in enumerate(captured):
            assert_same_bits(
                replayed_bits(result), expected[name], (replay, name, index)
            )


@skipIfNotCUDA()
def test_a_grid_of_other_ctas_than_the_kernel_was_bound_for_keeps_every_bit():
    """A persistent partition reads the grid it was launched with: one CTA
    runs every unit whole and its spare warp is asked for no look; a few
    CTAs each run a part of most units, many of them in the middle of a
    unit in three parts or more, which post and look for nothing. Each
    launch stores the bits of the kernel whose group looks, on the same
    grid."""
    args, off, on = _pair(SPARE, BALANCED)
    units = cast("torch.Tensor", args[6]).numel()
    patterns = stream_patterns(units, PAGES * PAGE - 1)
    for grid in (1, 2, 7, 47, 60):
        for name in ("ragged", "even empty", "short"):
            set_request_lengths(args, patterns[name])
            expected = poisoned_launch(
                lambda grid=grid: launch_with_grid(off, args, grid)
            )
            actual = poisoned_launch(lambda grid=grid: launch_with_grid(on, args, grid))
            assert_same_bits(actual, expected, (grid, name))


# One compile pair each: a skewed partition (the unit's other parts are
# found by the walk over CTAs), the seeded combination and its form under
# the store hint, the look beside the range-ahead role the same warp runs
# first, V issued with its K, sleeping waits, a short window, the reference
# recurrence, five units to four CTAs, the wider panels, whose stage has a
# row a lane (32 columns) and two rows for eight lanes (40) and whose
# other part's panel lies in the raw K ring whole (32) or but for its last
# eight columns (40), and a kernel whose threads all join at the body's
# end, where the K conversions and the spare warp wait behind their
# handover of that ring.
SCHEDULES = {
    "skew1": (SPARE, {**BALANCED, SKEW: 1}, {}),
    "seeded": (SPARE, {**WALK, AHEAD: 48, TILES: 5}, {}),
    "seeded-store-hint": (
        SPARE,
        {**WALK, AHEAD: 96, TILES: 5, "cute_store_policy": "l2_evict_last"},
        {"ctas": 100},
    ),
    "range-ahead": (SPARE, {**BALANCED, AHEAD: 48}, {}),
    "lag0": (SPARE, {**BALANCED, V_LAG: 0}, {}),
    "lag3-sleep": (SPARE, {**BALANCED, V_LAG: 3, "cute_mma_dag_sleep_waits": True}, {}),
    "short-window": (SPARE, WALK, {"window": SHORT_WINDOW}),
    "reference": (SPARE, BALANCED, {"reference_scores": True}),
    "ctas-100": (SPARE, BALANCED, {"ctas": 100}),
    "columns-32": (SPARE, WALK, {"rows": 32}),
    "columns-40": (SPARE, BALANCED, {"rows": 40}),
    "columns-40-tiles": (SPARE, {**WALK, TILES: 5}, {"rows": 40}),
    "no-early-retirement": (
        SPARE,
        {**BALANCED, "cute_mma_dag_early_tmem_retirement": False},
        {"rows": 40},
    ),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_the_look_keeps_every_bit_across_schedule_knobs(schedule):
    candidate, knobs, options = SCHEDULES[schedule]
    _check(candidate, knobs, calls=2, **options)
