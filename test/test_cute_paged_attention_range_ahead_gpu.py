"""The range-ahead prefetch of the balanced partition changes no output
bit (``cute_mma_dag_boundary_prefetch_ahead``): alone, beside a raw V ring
of five tiles and beside the reversed walk the seeds pair the two with;
eager and as captured calls that overlap their predecessors, over units of
unequal length that the balanced partition splits between CTAs, on a
skewed partition and on grids of fewer active CTAs than the distance
reaches. And a page table that the launch ahead stores is read behind those
stores: the role reads page ids at its CTA's start, ahead of every other
read of the kernel."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import MULTIWAVE_LAST_WAVE
from helion._testing.cute_capture_replay import compile_multiwave_defaults
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_folded_attention import folded_paged_attention
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import launch_with_grid
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_stream import PERSISTENT_CTAS
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import check_persistent_bits
from helion._testing.cute_paged_stream import device_shared_memory
from helion._testing.cute_paged_stream import persistent_pair
from helion._testing.cute_paged_stream import stream_patterns
from helion._testing.poisoned_outputs import poisoned_launch
from helion.runtime.cute import capture_pdl
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = "cute_mma_dag_boundary_prefetch_ahead"
TILES = "cute_mma_dag_value_ring_tiles"
PARTITION = "cute_mma_dag_partition"
SKEW = "cute_mma_dag_partition_skew"
ORDER = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
BALANCED = {PARTITION: "balanced"}
# The walk the seeds pair the two options with.
WALK = {**BALANCED, ORDER: "reversed", LEAD: 8}
# CTAs of the persistent launch: 64 have the distance of 48 alone, 100 both.
NEAR_CTAS = PERSISTENT_CTAS
FAR_CTAS = 100


def _rendered(candidate: dict[str, object]) -> dict[str, str]:
    """The paired body's keywords for ``candidate``'s options: the distance
    under its own name, and a tile count on the capacity that admitted it."""
    rendered = {"boundary_prefetch_ahead": str(candidate[KEY])}
    if TILES in candidate:
        rendered["value_ring_tiles"] = str(candidate[TILES])
        rendered["shared_memory_capacity"] = str(device_shared_memory())
    return rendered


def _check(
    candidate: dict[str, object],
    ctas: int,
    *,
    window: int = PAGES * PAGE - 1,
    rows: int = 8,
    calls: int = 3,
    **schedule: object,
) -> None:
    check_persistent_bits(
        candidate,
        ctas,
        _rendered(candidate),
        window=window,
        rows=rows,
        calls=calls,
        **schedule,
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("distance,ctas", [(48, NEAR_CTAS), (96, FAR_CTAS)])
def test_a_range_ahead_prefetch_keeps_every_output_bit(distance, ctas):
    """Units of zero to seventeen pages, two to a CTA or five to four CTAs:
    the served CTA's range starts inside a unit, on a unit's header or on a
    unit without pages, and its first unit has from no step to seventeen
    in the range. The CTAs within the distance of the grid's end serve
    nobody."""
    _check({KEY: distance}, ctas, **BALANCED)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("distance,ctas", [(48, NEAR_CTAS), (96, FAR_CTAS)])
def test_the_seeded_combination_keeps_every_output_bit(distance, ctas):
    """The walk newest first with the clipped page eight pages from a
    unit's end, the range-ahead prefetch and the ring of five tiles: what
    the seeds name together. Under the walk the served loader's first
    request is its unit's newest page, which is the page the role requests
    for it first."""
    _check({KEY: distance, TILES: 5}, ctas, **WALK)


@skipIfNotCUDA()
def test_a_grid_of_fewer_ctas_than_the_distance_reaches_serves_what_it_has():
    """A persistent partition reads the grid it was launched with. Launched
    with fewer CTAs than the distance, no CTA of the range-ahead kernel
    serves another; with a few more, only the first ones do; with one CTA
    it runs every unit. Each launch stores the bits of the kernel without
    the role on the same grid."""
    candidate: dict[str, object] = {KEY: 48}
    args, off, on = persistent_pair(
        candidate, NEAR_CTAS, _rendered(candidate), **BALANCED
    )
    units = cast("torch.Tensor", args[6]).numel()
    patterns = stream_patterns(units, PAGES * PAGE - 1)
    for grid in (1, 7, 47, 48, 49, 60):
        for name in ("ragged", "even empty", "short"):
            set_request_lengths(args, patterns[name])
            expected = poisoned_launch(
                lambda grid=grid: launch_with_grid(off, args, grid)
            )
            actual = poisoned_launch(lambda grid=grid: launch_with_grid(on, args, grid))
            assert_same_bits(actual, expected, (grid, name))


# One compile pair each: a skewed partition (the served range is cut by the
# applied skew), a short window (a unit streams at most four pages, fewer
# than the role requests and than the ring holds, and its first tile is
# clipped), the role beside the default order and the combination under the
# store hint the seeds repeat it with, and 32 query columns.
SCHEDULES = {
    "skew1": ({KEY: 48}, NEAR_CTAS, {**BALANCED, SKEW: 1}, {}),
    "skew1-combination": ({KEY: 48, TILES: 5}, NEAR_CTAS, {**WALK, SKEW: 1}, {}),
    "short-window": ({KEY: 48, TILES: 5}, NEAR_CTAS, WALK, {"window": SHORT_WINDOW}),
    "short-window-far": ({KEY: 96}, FAR_CTAS, BALANCED, {"window": SHORT_WINDOW}),
    "clip-both": (
        {KEY: 48, TILES: 5},
        NEAR_CTAS,
        {**WALK, "cute_mma_dag_key_clip": "both"},
        {"window": SHORT_WINDOW},
    ),
    "store-hint": (
        {KEY: 96, TILES: 5},
        FAR_CTAS,
        {**WALK, "cute_store_policy": "l2_evict_last"},
        {},
    ),
    "columns-32": ({KEY: 48, TILES: 5}, NEAR_CTAS, WALK, {"rows": 32}),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_the_role_keeps_every_bit_across_schedule_knobs(schedule):
    candidate, ctas, knobs, options = SCHEDULES[schedule]
    _check(candidate, ctas, calls=2, **options, **knobs)


# The aliasing scenario: the folded fixture, whose output is an argument,
# with one query a request, on a grid of two waves.
ALIAS_TABLE = 64
ALIAS_PAGE_BYTES = 33792
ALIAS_POOL = 13
ALIAS_REPLAYS = 50


def _folded_bank(requests: int) -> list[object]:
    """The folded fixture's arguments for ``requests`` requests of one query
    each, a pool of ``ALIAS_POOL`` pages and a page table of ``ALIAS_TABLE``
    pages a request under a window that holds them."""
    q, k, v, ks, vs, _table, _lengths, _window, block, lse, out = folded_attention_args(
        1
    )

    def grown(tensor: object, leading: int, storage: torch.Tensor | None = None):
        like = cast("torch.Tensor", tensor)
        if storage is None:
            elements = like.storage_offset() + leading * like.stride(0)
            storage = torch.empty(elements, dtype=like.dtype, device=DEVICE)
        return storage.view(like.dtype).as_strided(
            (leading, *like.shape[1:]), like.stride(), like.storage_offset()
        )

    pools = [
        torch.empty(ALIAS_POOL * ALIAS_PAGE_BYTES, dtype=torch.int8, device=DEVICE)
        for _ in range(2)
    ]
    return [
        grown(q, requests),
        grown(k, ALIAS_POOL, pools[0]),
        grown(v, ALIAS_POOL, pools[1]),
        grown(ks, ALIAS_POOL, pools[0]),
        grown(vs, ALIAS_POOL, pools[1]),
        torch.empty((requests, ALIAS_TABLE), dtype=torch.int32, device=DEVICE),
        torch.empty(requests, dtype=torch.int32, device=DEVICE),
        ALIAS_TABLE * 128 - 1,
        block,
        lse,
        grown(out, requests),
    ]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    integer = torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
    return tensor.contiguous().view(integer).clone()


# What the second call of the aliasing scenario runs beside the seeded
# combination: nothing, or what the seeds of the spare warp's look add to
# it. That look reads and resets a counter of the launch's workspace and
# loads another part's row statistics from it, early in its CTA's life:
# behind the launch's dependencies too.
ALIAS_OPTIONS = {
    "combination": {},
    "look": {"cute_mma_dag_partition_look": "spare"},
}


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("options", list(ALIAS_OPTIONS))
def test_a_page_table_the_launch_ahead_stores_is_read_behind_its_stores(options):
    """Two captured calls back to back. The first stores its output over
    the second's page table: its V pool is zero, so it stores exact zeros,
    which are the valid page id zero. In program order the second call
    attends page zero wherever the first stored. The second call is the
    seeded combination, whose range-ahead role reads page ids at its CTA's
    start and whose loaders read them at every page: none of those reads
    may come ahead of the first call's stores. The capture guard sees that
    the second call loads what the first stores and has it wait at its
    entry, and every replay has the bits of the two calls run one after
    the other. Under ``look`` the second call also has its spare warp look
    for each unit's lowest part (``ALIAS_OPTIONS``).

    The first call runs one request a CTA on a grid of two waves and
    releases its dependents at every CTA's entry. Its first requests walk
    the whole table and every other request one page, so their CTAs store
    long after the SMs of the others are free, and the table lies on the
    rows they store. That is what gives the control its teeth: with the
    guard's decision for the second call overridden to the before-writes
    mode, which lets a launch read its inputs ahead of its dependency
    wait, replays are wrong (all fifty of them on a B200). The control asks
    for one: a dependent launch may start beside the launch ahead, but
    nothing has it do so in every replay.

    Each capture is made with a spare workspace of the balanced partition
    at hand (an eager launch makes one, a capture takes it). A capture
    without one zeroes the workspace's counters inside its graph, ahead of
    its first balanced launch: that launch then depends on the fill, starts
    when the launch ahead is complete whatever its mode, and the control
    would pass for a reason that is not the guard's. So both captures
    check that the second call depends on the first one alone.
    """
    if not is_cuda():
        pytest.skip("capture PDL requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    if sms <= 96:
        pytest.skip("the range-ahead distance of 96 needs more CTAs")
    requests = sms + MULTIWAVE_LAST_WAVE
    first, second = _folded_bank(requests), _folded_bank(requests)
    generator = torch.Generator().manual_seed(requests)
    for index, bank in enumerate((first, second)):
        q, k, v, ks, vs, table, lengths = (cast("torch.Tensor", x) for x in bank[:7])
        for storage in (k, v):
            backing = torch.empty(0, dtype=torch.int8, device=DEVICE).set_(
                storage.untyped_storage()
            )
            backing.copy_(
                torch.randint(
                    -8, 9, (backing.numel(),), generator=generator, dtype=torch.int8
                )
            )
        for scale in (ks, vs):
            scale.copy_(
                torch.randint(
                    125, 129, scale.shape, generator=generator, dtype=torch.uint8
                )
            )
        q.copy_(torch.randn(q.shape, generator=generator) / 4)
        table.copy_(
            (torch.arange(table.numel()) * 7 % (ALIAS_POOL - 1) + 1).reshape(
                table.shape
            )
        )
        if index == 0:
            # The launch ahead stores zeros: its V pool is zero. Its first
            # requests walk the whole table, the others one page.
            v.zero_()
            lengths.fill_(100)
            lengths[:MULTIWAVE_LAST_WAVE] = ALIAS_TABLE * 128
        else:
            lengths.copy_((torch.arange(requests) % 8 * 128 + 100).to(torch.int32))
    out = cast("torch.Tensor", first[10])
    # One buffer for the first call's output and the second call's page
    # table: the table's rows lie on the first requests' outputs.
    elements = out.storage_offset() + requests * out.stride(0)
    chain = torch.zeros(elements, dtype=torch.bfloat16, device=DEVICE)
    first[10] = chain.as_strided(out.shape, out.stride(), out.storage_offset())
    words = requests * ALIAS_TABLE
    aliased = chain.view(torch.int32)[:words].view(requests, ALIAS_TABLE)
    pristine = cast("torch.Tensor", second[5]).clone()
    second[5] = aliased
    # The bytes of the table lie on the outputs of the requests that store
    # last: the first ones, which walk the whole table.
    assert words * 4 <= MULTIWAVE_LAST_WAVE * out.stride(0) * 2
    bound = helion.kernel(
        folded_paged_attention.fn, backend="cute", static_shapes=True, fast_math=True
    ).bind(tuple(first))
    aliased.copy_(pristine)
    _normalized, _source, ahead, ahead_plain = compile_multiwave_defaults(bound, first)
    combination = helion.Config.from_dict(
        {
            "block_sizes": [8],
            "cute_mma_dag": "tcgen05",
            "cute_mma_dag_warps": 4,
            "cute_mma_dag_stages": 2,
            "cute_mma_dag_pass_warps": "attention_paired",
            "cute_mma_dag_query_tma": True,
            "cute_mma_dag_v_lag": 2,
            **WALK,
            KEY: 96,
            TILES: 5,
            **ALIAS_OPTIONS[options],
        }
    )
    keywords = body_keywords(bound.to_code(combination))
    assert keywords["boundary_prefetch_ahead"] == "96"
    assert keywords["value_ring_tiles"] == "5"
    assert ("partition_look" in keywords) is (options == "look")
    behind = bound.compile_config(combination)
    behind_plain = bound.compile_config(
        helion.Config.from_dict({**combination.config, "cute_pdl": False})
    )
    aliased.copy_(pristine)
    for kernel in (behind, behind_plain):
        kernel(*second)
    torch.cuda.synchronize()

    def launched(kernel: Callable[..., object], bank: list[object]):
        output, lse = cast("tuple[torch.Tensor, torch.Tensor]", kernel(*bank))
        torch.cuda.synchronize()
        return _bits(output), _bits(lse)

    # Program order: the first call, then the second over the table as the
    # first left it.
    aliased.copy_(pristine)
    reference_first = launched(ahead_plain, first)
    assert not bool(reference_first[0].any())
    stored = _bits(aliased)
    assert bool((stored != _bits(pristine)).any())
    reference = launched(behind_plain, second)
    # A second call that read the table as it was has other bits.
    aliased.copy_(pristine)
    stale = launched(behind_plain, second)
    assert not torch.equal(stale[0], reference[0])

    def replays(forced: bool) -> int:
        """Replays of the captured pair whose second call has other bits
        than program order's."""
        decide = capture_pdl.CaptureGuard.decide
        behind_the_first: list[bool] = []

        def overridden(self, key, node, dependencies, effects):
            decision = decide(self, key, node, dependencies, effects)
            if not dependencies:
                # The first call, which the capture starts with.
                return decision
            (dependency,) = dependencies
            behind_the_first.append(dependency.node in self.captures.get(key, ()))
            if not forced:
                return decision
            return dataclasses.replace(decision, mode=WaitMode.BEFORE_WRITES)

        # The eager launch that makes the capture's spare workspace.
        aliased.copy_(pristine)
        launched(behind, second)
        aliased.copy_(pristine)
        graph = torch.cuda.CUDAGraph()
        with (
            patch.object(capture_pdl.CaptureGuard, "decide", overridden),
            observed_wait_modes() as modes,
            torch.cuda.graph(graph),
        ):
            captured = [ahead(*first), behind(*second)]
        assert behind_the_first == [True]
        # The second call loads what the first stores, and waits at its
        # entry unless the control says otherwise.
        late = WaitMode.BEFORE_WRITES if forced else WaitMode.EARLY
        assert modes == [WaitMode.EARLY, late], modes
        wrong = 0
        for _replay in range(ALIAS_REPLAYS):
            aliased.copy_(pristine)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            output, lse = cast("tuple[torch.Tensor, torch.Tensor]", captured[1])
            wrong += not (
                torch.equal(_bits(output), reference[0])
                and torch.equal(_bits(lse), reference[1])
            )
        return wrong

    assert replays(forced=False) == 0
    # The control: the same pair with the second call allowed to read its
    # inputs ahead of its dependency wait. One wrong replay shows that the
    # scenario can race; that every replay does is the device's timing.
    assert replays(forced=True) > 0
