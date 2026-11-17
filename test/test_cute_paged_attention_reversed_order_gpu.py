"""The reversed page order walks each request's pages newest first, every
tile with its own mask and key clip, and its results stay within the
rounding of a reassociated recurrence of the source order's.

Three comparisons, on K pages that are zero or random under a request's
window, with one unit per CTA and on the persistent partitions of a grid
of more units than SMs. The kernel in reversed order with the clipped page last has the
bits of a kernel whose source reflects its own loop over the same inputs:
the walk is that reversal, tile for tile. A request whose walk is the
source's own (at most one page, or two under a lead of the clipped page)
has the source order's bits. Every other request is within ``ORDER_OUTPUT``
of the request's largest output and ``ORDER_LSE`` of its LSE, and within
the suite's bounds of the FP32 reference, eager and as captured
back-to-back launches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_paged_headwide import configure_headwide_case
from helion._testing.cute_paged_headwide import headwide_config
from helion._testing.cute_paged_headwide import headwide_kernel
from helion._testing.cute_paged_page_order import ORDER_LSE
from helion._testing.cute_paged_page_order import ORDER_OUTPUT
from helion._testing.cute_paged_page_order import check_reference
from helion._testing.cute_paged_page_order import host_args
from helion._testing.cute_paged_page_order import unit_page_range
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_partition_plan import active_ctas
from helion._testing.cute_paged_partition_plan import applied_skew
from helion._testing.cute_paged_partition_plan import reference_partition
from helion._testing.cute_paged_reversed_order import clipped_keys
from helion._testing.cute_paged_reversed_order import reversed_walk
from helion._testing.cute_paged_reversed_order import split_pool
from helion._testing.cute_paged_reversed_order import window_layouts
from helion._testing.cute_paged_stream import IDLE_SMS
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import launched_bits
from helion._testing.cute_paged_stream import one_wave_units
from helion._testing.cute_paged_stream import poison_results
from helion._testing.cute_paged_stream import replayed_bits
from helion._testing.cute_paged_stream import stream_pair
from helion._testing.poisoned_outputs import poison_
from helion._testing.poisoned_outputs import poisoned_empty
from helion._testing.poisoned_outputs import written_bits
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
RING = "cute_mma_dag_value_ring"
PREFETCH = "cute_mma_dag_boundary_prefetch"
CLIP = "cute_mma_dag_key_clip"
V_LAG = "cute_mma_dag_v_lag"
PARTITION = "cute_mma_dag_partition"
SKEW = "cute_mma_dag_partition_skew"
# A page table of 32 pages a request over a pool of 48 pages, half of them
# with zero K. A window of 1500 keys: a request longer than it streams
# twelve or thirteen pages from a page that moves with its length, more
# than the lead of eight, and a shorter one every page it has.
TABLE = 32
POOL = 48
WINDOW = 1500
# The newest pages of a request that hold random K in the layout between
# all zero and all random.
FRESH = 4


def _patterns(units: int, rows: int) -> dict[str, list[int]]:
    """Request lengths for the page table."""
    generator = torch.Generator().manual_seed(units + rows)
    limit = TABLE * PAGE
    query_len = rows // 8
    edge = WINDOW + query_len
    ragged = torch.randint(0, limit + 1, (units,), generator=generator).tolist()
    return {
        # Longer than the window: the oldest page is the window's edge, at
        # a key that moves with the length.
        "clipped": [
            edge + 130 + index * 37 % (limit - edge - 130) for index in range(units)
        ],
        # The first row's window begins at a page boundary: it sees every
        # key of the oldest page.
        "page-aligned": [
            edge + PAGE * (1 + index % ((limit - edge) // PAGE))
            for index in range(units)
        ],
        # No longer than the window: every page from the table's first.
        "inside the window": [
            index * edge // max(units - 1, 1) for index in range(units)
        ],
        # None to eight pages: at most the lead, whose step clamps at the
        # walk's first.
        "short": [max((index % 9) * PAGE - index % 3, 0) for index in range(units)],
        "ragged": ragged,
        "even empty": [
            0 if index % 2 == 0 else ragged[index] for index in range(units)
        ],
        "all empty": [0] * units,
    }


def _walks(args: tuple[object, ...], lead: int) -> list[list[int]]:
    """Each request's walk by the model."""
    host = host_args(args)
    units = cast("torch.Tensor", args[6]).numel()
    return [
        reversed_walk(*unit_page_range(host, request), lead) for request in range(units)
    ]


def _per_request(value: torch.Tensor, units: int) -> torch.Tensor:
    """A request's two heads as one row."""
    return value.reshape(units, -1)


def _compare(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    same: torch.Tensor,
    what: object,
) -> tuple[float, float]:
    """A request whose walk is the source's own, and one none of whose rows
    sees a key, has the source order's bits; every other one is within the
    bounds, and a row without a visible key has the source order's LSE.
    Returns the largest deviations."""
    units = same.numel()
    output = _per_request(actual[0].view(torch.bfloat16).float(), units)
    wanted = _per_request(expected[0].view(torch.bfloat16).float(), units)
    lse = _per_request(actual[1].view(torch.float32), units)
    wanted_lse = _per_request(expected[1].view(torch.float32), units)
    scale = wanted.abs().amax(dim=1)
    exact = same | (scale == 0)
    for value, reference in zip(actual, expected, strict=True):
        equal = (_per_request(value, units) == _per_request(reference, units)).all(
            dim=1
        )
        assert bool(equal[exact].all()), (what, (~equal & exact).nonzero().flatten())
    walked = ~exact
    if not bool(walked.any()):
        return 0.0, 0.0
    moved = (output - wanted).abs().amax(dim=1)
    assert bool((moved[walked] <= ORDER_OUTPUT * scale[walked]).all()), (
        what,
        (moved[walked] / scale[walked]).amax(),
    )
    seen = torch.isfinite(wanted_lse)
    assert torch.equal(
        _per_request(actual[1], units)[~seen], _per_request(expected[1], units)[~seen]
    ), what
    moved_lse = torch.where(seen, (lse - wanted_lse).abs(), 0).amax(dim=1)
    assert bool((moved_lse[walked] <= ORDER_LSE).all()), (
        what,
        moved_lse[walked].amax(),
    )
    return (
        (moved[walked] / scale[walked]).amax().item(),
        moved_lse[walked].amax().item(),
    )


def _reversed_pair(
    candidate: dict[str, object],
    schedule: dict[str, object],
    rows: int,
    **options: object,
) -> tuple[tuple[object, ...], object, object, int]:
    """The fixture's arguments on the test device, its kernel in source
    order under ``schedule``, the same kernel in reversed order with
    ``candidate``'s knobs as well, and that kernel's lead of the clipped
    page."""
    args, source, walked, plain, rendered = stream_pair(
        {KEY: "reversed", **candidate},
        schedule=schedule,
        window=WINDOW,
        rows=rows,
        table_pages=TABLE,
        pool_pages=POOL,
        **options,  # type: ignore[arg-type]
    )
    assert "page_order" not in plain and "clipped_page_lead" not in plain
    keywords = body_keywords(rendered)
    assert keywords["page_order"] == "'reversed'"
    assert "page_order_shift" not in keywords and "reflected_dtype" not in keywords
    lead = cast("int", candidate.get(LEAD, 0))
    assert keywords.get("clipped_page_lead", "0") == str(lead)
    if RING in candidate:
        # The reversed kernel under test is the one with that ring depth.
        assert keywords["value_ring"] == str(candidate[RING])
    for knobs in (schedule, candidate):
        if PREFETCH in knobs:
            assert keywords["boundary_prefetch"] == str(knobs[PREFETCH])
    if SKEW in schedule:
        # Both kernels cut their units' ranges under that skew.
        for code in (plain, rendered):
            assert body_keywords(code)["partition_skew"] == str(schedule[SKEW])
    return args, source, walked, lead


def _check(
    candidate: dict[str, object] | None = None,
    *,
    rows: int = 40,
    calls: int = 3,
    overlapped: bool = True,
    bounds: list[tuple[int, int]] | None = None,
    reference_scores: bool = False,
    reserved_sms: int = 0,
    **schedule: object,
) -> dict[str, int]:
    """Every pattern over every layout eager, then as captured back-to-back
    launches replayed over them. Returns how many requests each class of
    walk had: clipped (the window hides keys of the oldest page from the
    first row) with more pages than the lead, unclipped with more pages
    than it, and with at most the lead's pages; under a balanced partition
    of ``schedule`` also how many parts of units in several parts its CTAs
    ran, and how many of those had fewer steps than the lead, and under a
    range skew of ``schedule`` how many patterns' launches applied one.
    ``overlapped``: the schedule carries the guarded capture policy, and
    each captured call but the first overlaps the one ahead.
    ``reserved_sms`` shrinks the grid of a persistent partition."""
    args, source, walked, lead = _reversed_pair(
        candidate or {},
        schedule,
        rows,
        bounds=bounds is not None,
        reference_scores=reference_scores,
        reserved_sms=reserved_sms,
    )
    data = split_pool(args)
    table_ids = cast("torch.Tensor", args[5])
    units = cast("torch.Tensor", args[6]).numel()
    if bounds is not None:
        ranges = cast("torch.Tensor", args[10])
        ranges.copy_(
            ranges.new_tensor([bounds[index % len(bounds)] for index in range(units)])
        )
    patterns = _patterns(units, rows)
    original = table_ids.clone()
    expected = {}
    classes = {"clipped": 0, "unclipped": 0, "short": 0}
    balanced = schedule.get(PARTITION) == "balanced"
    if balanced:
        classes.update({"parts": 0, "parts shorter than the lead": 0})
    skew = cast("int", schedule.get(SKEW, 0))
    if skew:
        classes["patterns under a skew"] = 0
    grid = one_wave_units() + IDLE_SMS - reserved_sms
    worst = worst_lse = 0.0
    try:
        for name, values in patterns.items():
            set_request_lengths(args, values)
            walks = _walks(args, lead)
            host = host_args(args)
            same = torch.tensor(
                [
                    walk == list(range(*unit_page_range(host, request)))
                    for request, walk in enumerate(walks)
                ],
                device=table_ids.device,
            )
            for request, walk in enumerate(walks):
                if not walk:
                    continue
                if len(walk) <= max(lead, 1):
                    classes["short"] += 1
                elif bounds is None and clipped_keys(args, request):
                    classes["clipped"] += 1
                else:
                    classes["unclipped"] += 1
            if balanced:
                # The parts the launch's CTAs run, by the partition's model.
                pages = [len(walk) for walk in walks]
                for segments in reference_partition(pages, grid, "balanced", skew=skew):
                    for segment in segments:
                        if not segment.whole:
                            classes["parts"] += 1
                            classes["parts shorter than the lead"] += (
                                0 < segment.count < lead
                            )
                if skew:
                    # A launch applies at most a thirty-second of its mean
                    # share: none of it where the requests are short.
                    slots = sum(pages) + len(pages)
                    classes["patterns under a skew"] += (
                        applied_skew(slots, active_ctas(pages, grid), skew) > 0
                    )
            layouts = window_layouts(args, data, FRESH, units + len(expected))
            for layout, table in layouts.items():
                what = (name, layout)
                table_ids.copy_(table)
                wanted = launched_bits(source, args)
                actual = launched_bits(walked, args)
                expected[what] = (table, actual)
                moved, moved_lse = _compare(actual, wanted, same, what)
                worst, worst_lse = max(worst, moved), max(worst_lse, moved_lse)
                # The walk is a function of the inputs: the same launch
                # again has the same bits.
                assert_same_bits(launched_bits(walked, args), actual, (what, "again"))
                if bounds is None:
                    check_reference(args, actual, what)
                    check_reference(args, wanted, (what, "source"))
        graph = torch.cuda.CUDAGraph()
        with observed_wait_modes() as modes, torch.cuda.graph(graph):
            captured = [
                cast("Callable[..., object]", walked)(*args) for _ in range(calls)
            ]
        if overlapped:
            assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * (calls - 1)
        for name, values in patterns.items():
            set_request_lengths(args, values)
            for what, (table, bits) in expected.items():
                if what[0] != name:
                    continue
                table_ids.copy_(table)
                poison_results(captured)
                graph.replay()
                torch.cuda.synchronize()
                for index, result in enumerate(captured):
                    assert_same_bits(replayed_bits(result), bits, (what, index))
    finally:
        table_ids.copy_(original)
    print(
        f"reversed order, lead {lead}: {classes}, largest output deviation "
        f"{worst:.5f} of a request's largest output, LSE {worst_lse:.2e}"
    )
    return classes


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_reversed_walk_has_the_bits_of_a_source_that_reflects_its_loop():
    """With the clipped page last the walk is the reversal of the source's
    loop, each step the tile its page has in source order: over the same
    inputs the kernel has the bits of the fixture whose source iterates its
    pages in reverse, in source order. A wrong page, mask or key clip at any
    step would differ."""
    args, _source, walked, lead = _reversed_pair({}, {}, 8)
    assert lead == 0
    mirror, reflected, _same, plain, _rendered = stream_pair(
        {},
        schedule={},
        window=WINDOW,
        reverse_pages=True,
        rows=8,
        table_pages=TABLE,
        pool_pages=POOL,
    )
    assert "reflected_dtype" in body_keywords(plain) and "page_order" not in plain
    data = split_pool(args)
    for index in range(5):
        cast("torch.Tensor", mirror[index]).copy_(cast("torch.Tensor", args[index]))
    table_ids = cast("torch.Tensor", args[5])
    units = cast("torch.Tensor", args[6]).numel()
    original = table_ids.clone()
    compared = 0
    try:
        for name, values in _patterns(units, 8).items():
            set_request_lengths(args, values)
            set_request_lengths(mirror, values)
            for layout, table in window_layouts(args, data, FRESH, units).items():
                table_ids.copy_(table)
                cast("torch.Tensor", mirror[5]).copy_(table)
                assert_same_bits(
                    launched_bits(walked, args),
                    launched_bits(reflected, mirror),
                    (name, layout),
                )
                compared += 1
    finally:
        table_ids.copy_(original)
    assert compared == 21


# The two forms a search is seeded with, on 40 query columns: the clipped
# page last beside the default ring, and eight pages ahead of the walk's end
# beside three pages of raw V per head and a boundary prefetch. Both run
# under the kernel's default key clip, which leaves the keys before every
# row's window out of the oldest tile's boxes.
FORMS = {
    "last-ring2": ({}, {}),
    "lead8-ring3-prefetch8": ({LEAD: 8, RING: 3}, {PREFETCH: 8}),
}


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("form", list(FORMS))
def test_the_reversed_walk_stays_within_rounding_of_the_source_order(form):
    """Requests longer than the window, whose oldest page the window clips
    or begins exactly at, requests inside the window and requests of at
    most the lead's pages, over windows of zero K, of data on the newest
    pages and of data everywhere."""
    candidate, schedule = FORMS[form]
    classes = _check(candidate, **schedule)
    # Every class of walk occurs: a clipped and an unclipped oldest page
    # ahead of the walk's end, and walks the lead does not fit in.
    assert all(count > 0 for count in classes.values()), classes


# One compile pair each: the lead beside the default ring and the deeper
# ring beside the page last (the seeds pair them the other way), a boundary
# prefetch of either seeded depth on the walk's first pages, whole tiles
# and a clip of both sides under the window, V issued with its K (no seed,
# a legal config), a lagged V with sleeping waits, the late retirement with
# its final join, one query row a request (every row sees the same keys),
# the register split the searched window configs carry, and the
# reference-score recurrence under the early probability schedule.
SCHEDULES: dict[str, tuple[dict[str, object], dict[str, object], dict[str, object]]] = {
    "lead8-ring2": ({LEAD: 8}, {}, {}),
    "last-ring3": ({RING: 3}, {}, {}),
    "last-prefetch8": ({}, {PREFETCH: 8}, {}),
    "lead8-ring3-prefetch12": ({LEAD: 8, RING: 3}, {PREFETCH: 12}, {}),
    "lead8-clip-both": ({LEAD: 8}, {CLIP: "both"}, {}),
    "lead8-clip-off": ({LEAD: 8}, {CLIP: "off"}, {}),
    "lead8-ring3-lag0": ({LEAD: 8, RING: 3}, {V_LAG: 0}, {}),
    "last-lag3-sleep": ({}, {V_LAG: 3, "cute_mma_dag_sleep_waits": True}, {}),
    "lead8-joined": ({LEAD: 8}, {"cute_mma_dag_early_tmem_retirement": False}, {}),
    "lead8-rows8": ({LEAD: 8}, {}, {"rows": 8}),
    "lead8-ring3-prefetch8-split": (
        {LEAD: 8, RING: 3},
        {PREFETCH: 8, "cute_mma_dag_register_quotas": "120/96/96/48"},
        {},
    ),
    "lead8-early-reference": (
        {LEAD: 8},
        {"cute_mma_dag_probability_schedule": "early"},
        {"reference_scores": True},
    ),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_the_reversed_walk_holds_across_schedule_knobs(schedule):
    candidate, knobs, options = SCHEDULES[schedule]
    classes = _check(candidate, **options, **knobs)
    assert all(count > 0 for count in classes.values()), classes


def _persistent(
    partition: str, candidate: dict[str, object], **schedule: object
) -> dict[str, int]:
    """``_check`` on a persistent partition of a grid with a third as many
    CTAs as units: every CTA streams three units or parts of them, the grid
    on which those partitions run the walk."""
    units = one_wave_units()
    return _check(
        candidate,
        rows=32,
        reserved_sms=units + IDLE_SMS - units // 3,
        **{PARTITION: partition},
        **schedule,
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_reversed_walk_stays_within_rounding_on_the_balanced_partition():
    """The form the balanced partition is seeded with, the clipped page
    eight pages ahead of a unit's end, against the same bind in source
    order: every unit of a CTA's stream is walked newest first, and a unit
    that several CTAs share is walked in parts, each with its steps of the
    unit's walk. Requests of every class of walk, units in parts, and parts
    of fewer steps than the lead, which hold the clipped page's step or do
    not."""
    classes = _persistent("balanced", {LEAD: 8})
    assert all(count > 0 for count in classes.values()), classes


# One compile pair each, on a persistent partition: whole units, which no
# CTA shares; the clipped page last; the walk under a range skew the launch
# applies, V issued with its K and the stores' L2 hint, as searched balanced
# configs carry them; and whole tiles under the window. The skew is two
# pages, the largest this bind offers (a third as many CTAs as units of at
# most 32 pages: mean shares of about a hundred slots), and the launches
# of its longer requests apply one page of it.
PERSISTENT = {
    "whole-lead8": ("whole", {LEAD: 8}, {}),
    "balanced-last": ("balanced", {}, {}),
    "balanced-lead8-lag0-hint": (
        "balanced",
        {LEAD: 8},
        {SKEW: 2, V_LAG: 0, "cute_store_policy": "l2_evict_last"},
    ),
    "balanced-lead8-clip-off": ("balanced", {LEAD: 8}, {CLIP: "off"}),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("form", list(PERSISTENT))
def test_the_reversed_walk_holds_on_the_persistent_partitions(form):
    partition, candidate, schedule = PERSISTENT[form]
    classes = _persistent(partition, candidate, **schedule)
    if candidate.get(LEAD):
        assert all(count > 0 for count in classes.values()), classes
    else:
        # No part is shorter than a lead the walk does not have.
        assert not classes.pop("parts shorter than the lead")
        assert all(count > 0 for count in classes.values()), classes


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("lead", [0, 8])
def test_tiles_no_row_sees_keep_their_masks_in_the_walk(lead):
    """Page ranges from a tensor, wider than the keys a request's rows see:
    the walk's first tiles lie past the request's length and its last ones
    before its window, all masked, and the page the lead moves is the
    range's oldest, not the window's edge. Every tile keeps the mask of its
    own page, so the results are the source order's within rounding."""
    classes = _check(
        {LEAD: lead} if lead else {},
        bounds=[(0, TABLE), (3, TABLE - 2), (0, 20)],
    )
    assert classes["unclipped"] > 0


# The recurrence the walk was measured on: scores in base two against a
# running maximum that a whole head raises together and only past a
# threshold, normalized by a reciprocal (the head-wide fixture). Its page
# table has 20 pages a request over a pool of 24, under a window of 2047
# keys, sixteen pages.
HEADWIDE_TABLE = 20
HEADWIDE_POOL = 24
HEADWIDE_WINDOW = 2047
# The lengths of the fixture's two requests: both longer than the window
# (seventeen pages, the oldest clipped); one whose first row's window begins
# at a page boundary and one inside the window; eight pages, which a lead
# of eight takes oldest first, and three; one page, whose walk is the
# source's, and none.
HEADWIDE_LENGTHS = (
    (2560, 2190),
    (HEADWIDE_WINDOW + 5 + 2 * PAGE, 1301),
    (1021, 257),
    (PAGE, 0),
)
# The fixture's finite inputs: its own keys, a negated Q over other pages,
# zero K, a maximum below zero, and heads whose page maxima fall from page
# to page beside heads that keep theirs, which votes differently from one
# order to the other.
HEADWIDE_CASES = (
    "original",
    "changed_q",
    "zero_maximum",
    "negative_maximum",
    "mixed_rows",
)


def _headwide_bits(kernel: object, args: tuple[object, ...]) -> list[torch.Tensor]:
    """The bits one eager call stores into the fixture's padded output and
    its LSE, both started as the payload."""
    poison_(cast("torch.Tensor", args[10]))
    with poisoned_empty():
        results = cast("Callable[..., tuple[torch.Tensor, ...]]", kernel)(*args)
    return [written_bits(value) for value in results]


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_walk_holds_on_the_headwide_thresholded_recurrence():
    """The head-wide fixture's source reflects its loop and is not offered
    the walk; the same source over a loop that visits its pages oldest
    first is. In reversed order with the clipped page last that kernel has
    the bits of the reflecting fixture, launch for launch, and with either
    lead its requests stay within the module's bounds of its own source
    order."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    query_len = 5
    values = list(
        folded_attention_args(
            query_len,
            device=DEVICE,
            pages=HEADWIDE_POOL,
            table_pages=HEADWIDE_TABLE,
        )
    )
    values[7] = HEADWIDE_WINDOW
    args = tuple(values)
    configure_headwide_case(args, "original")
    oldest_first = headwide_kernel(reverse_pages=False).bind(args)
    reflecting = headwide_kernel().bind(args)

    def config(**knobs: object):
        # The walk has no value-metadata transport.
        config = headwide_config(query_len)
        config.config.update(cute_mma_dag_value_metadata_skip=False, **knobs)
        return config

    plain = body_keywords(oldest_first.to_code(config()))
    assert plain["headwide_rescale"] == "True"
    assert plain["rowwise_rescale_threshold"] == "2.0"
    assert "page_order" not in plain and "reflected_dtype" not in plain
    assert "reflected_dtype" in body_keywords(reflecting.to_code(config()))
    source = oldest_first.compile_config(config())
    mirror = reflecting.compile_config(config())
    walked = {}
    for lead in (0, 8):
        knobs: dict[str, object] = {KEY: "reversed"}
        if lead:
            knobs[LEAD] = lead
        keywords = body_keywords(oldest_first.to_code(config(**knobs)))
        assert keywords["page_order"] == "'reversed'"
        assert keywords.get("clipped_page_lead", "0") == str(lead)
        assert keywords["headwide_rescale"] == "True"
        walked[lead] = oldest_first.compile_config(config(**knobs))
    lengths = cast("torch.Tensor", args[6])
    # No request is exempt: one whose walk is the source's own, or that
    # sees no key, is within the bounds as well.
    none_same = torch.zeros(lengths.numel(), dtype=torch.bool, device=lengths.device)
    worst = worst_lse = 0.0
    for case in HEADWIDE_CASES:
        for pair in HEADWIDE_LENGTHS:
            configure_headwide_case(args, case)
            lengths.copy_(lengths.new_tensor(pair))
            what = (case, pair)
            wanted = _headwide_bits(source, args)
            assert_same_bits(
                _headwide_bits(walked[0], args),
                _headwide_bits(mirror, args),
                (what, "the reflecting source"),
            )
            for lead, kernel in walked.items():
                actual = _headwide_bits(kernel, args)
                moved, moved_lse = _compare(actual, wanted, none_same, (what, lead))
                worst, worst_lse = max(worst, moved), max(worst_lse, moved_lse)
                assert_same_bits(
                    _headwide_bits(kernel, args), actual, (what, lead, "again")
                )
    print(
        f"reversed order on the head-wide recurrence: largest output deviation "
        f"{worst:.5f} of a request's largest output, LSE {worst_lse:.2e}"
    )
    # The orders differ somewhere: the comparison is not of one kernel.
    assert worst > 0.0
