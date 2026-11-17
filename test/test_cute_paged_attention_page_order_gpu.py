"""The bucketed page order streams exactly the modelled permutation of each
request's interior pages and nothing else, and its results stay within the
rounding of a reassociated recurrence of the source order's.

Three comparisons, on random K, V and page tables. The kernel in bucketed
order over a page table has the bits of the kernel in source order over the
table the model reorders (``bucketed_page_table``): the device permutes the
interior run the definitions name, by a stable sort, and no other tile. A
request the model does not reorder has the source order's bits. A request
it does reorder is within ``ORDER_OUTPUT`` of the request's largest output
and ``ORDER_LSE`` of its LSE, and within the suite's bounds of the FP32
reference.

A request of no more pages than the head has no step that could take the
order, and its loader does not wait for the sorting warp: such requests are
launched under the schedules whose body ends with its all-thread join."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_HEAD
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_MIN_SORTED
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_TAIL
from helion._compiler.cute.paged_attention_page_order import page_order_shift
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_paged_page_order import ORDER_LSE
from helion._testing.cute_paged_page_order import ORDER_OUTPUT
from helion._testing.cute_paged_page_order import bucketed_page_table
from helion._testing.cute_paged_page_order import check_reference
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import launched_bits
from helion._testing.cute_paged_stream import poison_results
from helion._testing.cute_paged_stream import replayed_bits
from helion._testing.cute_paged_stream import stream_pair
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = "cute_mma_dag_page_order"
RING = "cute_mma_dag_value_ring"
V_LAG = "cute_mma_dag_v_lag"
# A page table of 64 pages a request over a pool of 80 pages: a request has
# more interior pages behind the head and ahead of the tail than the fewest
# a sort orders, and a bucket is four pages of the pool.
TABLE = 64
POOL = 80
SHIFT = page_order_shift(POOL)
# The fewest pages of a request that is reordered: its head, its tail, of
# which the last page is a boundary tile, and the fewest interior pages a
# sort orders between them.
FEWEST = PAGE_ORDER_HEAD + PAGE_ORDER_TAIL + PAGE_ORDER_MIN_SORTED
assert FEWEST + 2 <= TABLE
# A window of fifty pages: a long request starts at a later page, whose
# leading keys the kernel's default clip leaves out, and still has enough
# interior pages behind it and its head for a sort.
LONG_WINDOW = 50 * PAGE
assert 50 + 1 - PAGE_ORDER_HEAD - PAGE_ORDER_TAIL >= PAGE_ORDER_MIN_SORTED


def _patterns(units: int, window: int, table: int = TABLE) -> dict[str, list[int]]:
    """Request lengths for a page table of ``table`` pages."""
    generator = torch.Generator().manual_seed(units + window)
    limit = table * PAGE
    ragged = torch.randint(0, limit + 1, (units,), generator=generator).tolist()
    return {
        "full": [limit] * units,
        "ragged": ragged,
        "even empty": [
            0 if index % 2 == 0 else ragged[index] for index in range(units)
        ],
        # Fewer pages than the tail keeps in source order: nothing to sort.
        "below the tail": torch.randint(
            0, PAGE_ORDER_TAIL * PAGE, (units,), generator=generator
        ).tolist(),
        # One page fewer than the fewest a request is reordered for, up to
        # two pages more: 31 to 34 interior pages behind the head and ahead
        # of the tail.
        "threshold": [
            (FEWEST - 1 + index % 4) * PAGE - index % 3 for index in range(units)
        ],
        # Lengths at and next to page boundaries, in every page of the table.
        "page edges": [
            (index % table + 1) * PAGE - index % 3 for index in range(units)
        ],
        "all empty": [0] * units,
    }


def _requests(output: torch.Tensor, request: int) -> torch.Tensor:
    """The rows of a request's two heads."""
    return output[2 * request : 2 * request + 2]


def _compare(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    reordered: list[int],
    units: int,
    what: object,
) -> tuple[float, float]:
    """A request the model does not reorder has the source order's bits; a
    reordered one is within the bounds. Returns the largest deviations."""
    output, lse = actual[0].view(torch.bfloat16).float(), actual[1].view(torch.float32)
    wanted, wanted_lse = (
        expected[0].view(torch.bfloat16).float(),
        expected[1].view(torch.float32),
    )
    worst = worst_lse = 0.0
    for request in range(units):
        if request not in reordered:
            for value, reference in zip(actual, expected, strict=True):
                assert torch.equal(
                    _requests(value, request), _requests(reference, request)
                ), (what, request)
            continue
        scale = _requests(wanted, request).abs().amax().item()
        assert scale > 0 and torch.isfinite(_requests(wanted_lse, request)).all()
        moved = (_requests(output, request) - _requests(wanted, request)).abs().amax()
        assert moved.item() <= ORDER_OUTPUT * scale, (what, request, moved / scale)
        worst = max(worst, moved.item() / scale)
        moved = (_requests(lse, request) - _requests(wanted_lse, request)).abs().amax()
        assert moved.item() <= ORDER_LSE, (what, request, moved)
        worst_lse = max(worst_lse, moved.item())
    return worst, worst_lse


def _check(
    window: int,
    *,
    calls: int = 2,
    bounds: list[tuple[int, int]] | None = None,
    reverse_pages: bool = False,
    table: int = TABLE,
    lengths: dict[str, list[int]] | None = None,
    candidate: dict[str, object] | None = None,
    overlapped: bool = True,
    **schedule: object,
) -> int:
    """Every pattern eager, then as captured back-to-back launches replayed
    over the patterns (``lengths``, or the table's own). Returns the
    requests that were reordered. ``overlapped``: the schedule carries the
    guarded capture policy, and each captured call but the first overlaps
    the one ahead."""
    args, source, bucketed, plain, rendered = stream_pair(
        {KEY: "bucketed", **(candidate or {})},
        schedule=schedule,
        window=window,
        reverse_pages=reverse_pages,
        table_pages=table,
        pool_pages=POOL,
        bounds=bounds is not None,
    )
    assert "page_order" not in plain
    keywords = body_keywords(rendered)
    assert keywords["page_order"] == "'bucketed'"
    assert keywords["page_order_shift"] == str(SHIFT)
    if candidate is not None and RING in candidate:
        # The bucketed kernel under test is the one with that ring depth.
        assert keywords["value_ring"] == str(candidate[RING])
    table_ids = cast("torch.Tensor", args[5])
    units = cast("torch.Tensor", args[6]).numel()
    if bounds is not None:
        ranges = cast("torch.Tensor", args[10])
        ranges.copy_(
            ranges.new_tensor([bounds[index % len(bounds)] for index in range(units)])
        )
    patterns = _patterns(units, window, table) if lengths is None else lengths
    original = table_ids.clone()
    expected = {}
    total = 0
    worst = worst_lse = 0.0
    try:
        for name, values in patterns.items():
            set_request_lengths(args, values)
            wanted = launched_bits(source, args)
            actual = launched_bits(bucketed, args)
            expected[name] = actual
            modelled, reordered = bucketed_page_table(args, SHIFT)
            total += len(reordered)
            moved, moved_lse = _compare(actual, wanted, reordered, units, name)
            worst, worst_lse = max(worst, moved), max(worst_lse, moved_lse)
            # The device streams exactly the model's permutation: in source
            # order over the reordered table the kernel has the same bits.
            table_ids.copy_(modelled)
            assert_same_bits(launched_bits(source, args), actual, (name, "model"))
            table_ids.copy_(original)
            # The order is a function of the inputs: the same launch again
            # has the same bits.
            assert_same_bits(launched_bits(bucketed, args), actual, (name, "again"))
            if (
                name in ("full", "ragged")
                and bounds is None
                and window == NO_WINDOW
                and not reverse_pages
            ):
                check_reference(args, actual, name)
        graph = torch.cuda.CUDAGraph()
        with observed_wait_modes() as modes, torch.cuda.graph(graph):
            captured = [
                cast("Callable[..., object]", bucketed)(*args) for _ in range(calls)
            ]
        if overlapped:
            assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * (calls - 1)
        for name, values in patterns.items():
            set_request_lengths(args, values)
            poison_results(captured)
            graph.replay()
            torch.cuda.synchronize()
            for index, result in enumerate(captured):
                assert_same_bits(replayed_bits(result), expected[name], (name, index))
    finally:
        table_ids.copy_(original)
    print(
        f"page order: {total} reordered requests, largest output deviation "
        f"{worst:.5f} of a request's largest output, LSE {worst_lse:.2e}"
    )
    return total


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_bucketed_order_permutes_interior_pages_within_rounding():
    """A kernel without a window: requests of every length, among them lists
    shorter than the tail and lists one page on either side of the fewest a
    request is reordered for."""
    args, _source, _bucketed, _plain, _rendered = stream_pair(
        {KEY: "bucketed"},
        schedule={},
        window=NO_WINDOW,
        table_pages=TABLE,
        pool_pages=POOL,
    )
    units = cast("torch.Tensor", args[6]).numel()
    patterns = _patterns(units, NO_WINDOW)
    # What the model reorders, by pattern: every full request, no request
    # shorter than the tail, and the requests at or above the minimum.
    counts = {}
    for name, values in patterns.items():
        set_request_lengths(args, values)
        counts[name] = len(bucketed_page_table(args, SHIFT)[1])
    assert counts["full"] == units
    assert counts["below the tail"] == counts["all empty"] == 0
    assert 0 < counts["ragged"] < units
    # In turn one page too few, the fewest and up to two more: 31 interior
    # candidates are too few, 32 enough.
    assert PAGE_ORDER_MIN_SORTED == 32
    assert counts["threshold"] == units - (units + 3) // 4
    assert _check(NO_WINDOW) == sum(counts.values())


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_clipped_window_edge_keeps_its_place():
    """A window of fifty pages under the kernel's default key clip: a long
    request's first tile is a window edge, clipped, and the interior run
    starts behind it. The bucketed kernel also stages three pages of raw V
    per head: the ordered loader's two halves and the V conversion warps
    must walk that ring at one depth, or the stream stalls or converts
    another page's tile."""
    assert _check(LONG_WINDOW, candidate={RING: 3}) > 0


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_tiles_no_row_sees_are_not_reordered():
    """Page ranges from a tensor, longer than the keys a request has: the
    tiles past a request's length are fully masked, lie among the sort's
    candidates and must keep their pages. A device that reordered them
    would stream pages into the visible steps that no row may see."""
    assert _check(NO_WINDOW, bounds=[(0, TABLE), (3, TABLE - 2), (0, 40)]) > 0


# The schedules whose body ends with its all-thread join: the late
# retirement, and the release behind the body (which is no capture overlap).
JOINED = {
    "late-retirement": ({"cute_mma_dag_early_tmem_retirement": False}, True),
    "exit-release": ({"cute_pdl": True, "cute_pdl_release": "exit"}, False),
}


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("joined", list(JOINED))
def test_a_unit_within_the_head_does_not_wait_for_the_sorting_warp(joined):
    """Requests of at most the head's pages with V issued with its K: the
    loader has issued its last request before the point at which a longer
    request's would wait for the sorting warp, and reaches the body's final
    join without that warp. Every such request has the source order's bits,
    eager and as captured back-to-back calls, also where each CTA's
    neighbours run requests long enough to be reordered."""
    knobs, overlapped = JOINED[joined]
    args, _source, _bucketed, _plain, rendered = stream_pair(
        {KEY: "bucketed"},
        schedule={V_LAG: 0, **knobs},
        window=NO_WINDOW,
        table_pages=TABLE,
        pool_pages=POOL,
    )
    keywords = body_keywords(rendered)
    assert "v_lag" not in keywords
    # Both forms end with the join: the late retirement by itself, the
    # release behind the body by keeping one common end.
    assert ("join_before_exit" in keywords) is (joined == "exit-release")
    assert ("early_tmem_retirement" in keywords) is (joined == "exit-release")
    units = cast("torch.Tensor", args[6]).numel()
    lengths = {
        # Zero to four pages, every third request a few keys short of them.
        "within the head": [
            max((index % (PAGE_ORDER_HEAD + 1)) * PAGE - index % 3, 0)
            for index in range(units)
        ],
        "the head": [PAGE_ORDER_HEAD * PAGE] * units,
        "one page": [PAGE] * units,
        "all empty": [0] * units,
        # Every other request the whole table: its loader waits for its
        # CTA's sorting warp beside neighbours that do not.
        "beside long ones": [
            TABLE * PAGE if index % 2 else (index % (PAGE_ORDER_HEAD + 1)) * PAGE
            for index in range(units)
        ],
    }
    reordered = _check(
        NO_WINDOW,
        calls=3,
        lengths=lengths,
        overlapped=overlapped,
        **{V_LAG: 0, **knobs},
    )
    # Only the long requests of the last pattern are reordered.
    assert reordered == units // 2


# One compile pair each: the V lags (a lagged V tile keeps the page its K
# tile requested), sleeping waits, the seeds' pairing with the deeper raw V
# ring and the longer lead of the tail release, a boundary prefetch behind
# the sort on the same warp, whole tiles and a clip of both sides under the
# window, the late retirement with its final join (which the sorting warp's
# lanes reach from their own paths through the sort), pages iterated in
# reverse, and another register split.
SCHEDULES = {
    "lag0": (NO_WINDOW, {"cute_mma_dag_v_lag": 0}, {}),
    "lag3-sleep": (
        NO_WINDOW,
        {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True},
        {},
    ),
    "ring3-lead16": (
        NO_WINDOW,
        {"cute_mma_dag_v_lag": 0},
        {
            "candidate": {
                "cute_mma_dag_value_ring": 3,
                "cute_mma_dag_release_lead": 16,
            }
        },
    ),
    "prefetch8": (NO_WINDOW, {"cute_mma_dag_boundary_prefetch": 8}, {}),
    "clip-both": (LONG_WINDOW, {"cute_mma_dag_key_clip": "both"}, {}),
    "clip-off": (LONG_WINDOW, {"cute_mma_dag_key_clip": "off"}, {}),
    "joined": (NO_WINDOW, {"cute_mma_dag_early_tmem_retirement": False}, {}),
    "reversed": (NO_WINDOW, {}, {"reverse_pages": True}),
    "split-120-96": (
        NO_WINDOW,
        {"cute_mma_dag_register_quotas": "120/96/96/48"},
        {},
    ),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_the_bucketed_order_holds_across_schedule_knobs(schedule):
    window, knobs, options = SCHEDULES[schedule]
    assert _check(window, **options, **knobs) > 0


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_run_may_start_at_any_lane_of_the_sorting_warp():
    """Pages iterated in reverse over a range from a tensor: a request's
    first steps are the tiles past its length, which no row sees, so its
    interior run starts in mid-stream. With 96 pages in the range and
    lengths of 53 to 68 pages the run's first candidate is each of 24 to
    40, the warp's last lane and the next round's first among them."""
    table = 96
    units = cast(
        "torch.Tensor",
        stream_pair(
            {KEY: "bucketed"},
            schedule={},
            window=NO_WINDOW,
            reverse_pages=True,
            table_pages=table,
            pool_pages=POOL,
            bounds=True,
        )[0][6],
    ).numel()
    lengths = {
        "mid-stream": [(53 + index % 16) * PAGE - index % 3 for index in range(units)]
    }
    reordered = _check(
        NO_WINDOW,
        bounds=[(0, table)],
        reverse_pages=True,
        table=table,
        lengths=lengths,
    )
    assert reordered == units
