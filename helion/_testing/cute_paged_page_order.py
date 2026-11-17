"""Reference model and lane mirror of the paired schedule's bucketed page
order.

``ordered_run`` states which stream steps of a request the order permutes
and what each then streams, straight from the definitions: the fixture
source's page range and mask, the order's head, tail and capacity, and a
stable sort by bucket. ``bucketed_page_table`` writes that into a page
table, so the kernel in source order over the result streams what the
kernel in bucketed order streams over the original. ``mirror_sort``
repeats, lane by lane, what the device code in
``helion._compiler.cute._paged_attention_page_order`` computes, and
``run_device_sort`` runs that code itself on the host, 32 lanes in
lockstep, so CPU tests compare all three.

``ORDER_OUTPUT``, ``ORDER_LSE`` and ``check_reference`` are what the GPU
tests of both page orders, the bucketed one and the reversed walk, bound a
request's results by.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

from .cute_paged_attention_schedule import paged_attention_reference
from .cute_paged_partition import PAGE
from .cute_paged_stream import jit_host_body
from helion._compiler.cute import _paged_attention_compute as compute
from helion._compiler.cute import _paged_attention_page_order as device
from helion._compiler.cute import _paged_attention_partial_pipeline as partial
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_BUCKETS
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_CAPACITY
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_HEAD
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_MIN_SORTED
from helion._compiler.cute.paged_attention_page_order import PAGE_ORDER_TAIL
from helion._compiler.cute.paged_attention_page_order import page_order_bucket

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

# How far a request whose pages are streamed in another order may be from
# the same request in source order: a fraction of the request's largest
# output, and absolutely in its LSE. Both orders round each probability to
# BF16 against the running maximum and add the tiles' products in stream
# order; another order rounds at other points. Measured on the GPU fixtures
# of the bucketed order (thirteen kernels, their length patterns, 3,900
# reordered requests of 32 to 56 reordered pages): 0.0073 and 1.9e-6. On
# those of the reversed walk (sixteen kernels, their seven length patterns
# over three K layouts): 0.0072 and 1.9e-6.
ORDER_OUTPUT = 0.016
ORDER_LSE = 1e-5
# Either order against the FP32 reference, as the partition tests bound a
# unit: a fraction of the request's largest output, and its LSE.
REFERENCE_OUTPUT = 0.05
REFERENCE_LSE = 0.002
# Requests the reference is evaluated for: it walks every tile in Python.
REFERENCE_REQUESTS = 3


def check_reference(
    args: tuple[object, ...], actual: list[torch.Tensor], what: object
) -> None:
    """The first requests of a launch's stored bits ``actual`` (the BF16
    output and the FP32 LSE) against the FP32 reference of the fixture's
    source, which derives a request's pages from its length. A row that
    sees no key has the reference's LSE exactly."""
    count = REFERENCE_REQUESTS
    q, k, v, sk, sv, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    sub = (q[: 2 * count], k, v, sk, sv, pages[:count], lengths[:count], *args[7:10])
    exact, exact_lse = paged_attention_reference(sub, denominator_per_key=True)
    output = actual[0].view(torch.bfloat16).float()[: 2 * count]
    lse = actual[1].view(torch.float32)[: 2 * count]
    for request in range(count):
        if int(lengths[request].item()) == 0:
            continue
        rows = slice(2 * request, 2 * request + 2)
        wanted = exact.float()[rows]
        scale = wanted.abs().amax().item()
        error = (output[rows] - wanted).abs().amax().item()
        assert error <= REFERENCE_OUTPUT * scale, (what, request, error / scale)
        seen = torch.isfinite(exact_lse[rows])
        assert torch.equal(lse[rows][~seen], exact_lse[rows][~seen]), (what, request)
        error = torch.where(seen, (lse[rows] - exact_lse[rows]).abs(), 0).amax()
        assert error.item() <= REFERENCE_LSE, (what, request, error)


def query_columns(rows: int) -> int:
    """The paired panel's columns for ``rows`` query rows of an item."""
    return 8 if rows <= 8 else 32 if rows <= 32 else 40


def host_args(args: tuple[object, ...]) -> tuple[object, ...]:
    """``args`` with the page table, the lengths and the page ranges on the
    host: the models read them element by element."""
    return tuple(
        value.cpu()
        if isinstance(value, torch.Tensor) and index in (5, 6, 10)
        else value
        for index, value in enumerate(args)
    )


def unit_page_range(args: tuple[object, ...], request: int) -> tuple[int, int]:
    """The pages ``[first, last)`` the fixture's source visits for a request:
    the request's row of the bounds tensor, or the pages its length and
    window reach."""
    bounds = cast("torch.Tensor | None", args[10])
    if bounds is not None:
        first, last = (int(value) for value in bounds[request].tolist())
        return first, last
    length = int(cast("torch.Tensor", args[6])[request].item())
    query_len, window = cast("int", args[7]), cast("int", args[9])
    begin = max(length - query_len - window, 0) if window >= 0 else 0
    return begin // PAGE, (length + PAGE - 1) // PAGE


def stream_page(first: int, last: int, step: int, reverse: bool) -> int:
    """The logical page of a request's stream step ``step``."""
    return last - 1 - step if reverse else first + step


def interior_tile(args: tuple[object, ...], request: int, logical_page: int) -> bool:
    """Whether every key of the page is visible to every row of the panel:
    at most the first row's position and, under a window, at least the last
    row's position less the window. The rows are the panel's columns, as in
    the template's test, which a panel wider than the item's rows makes the
    stricter of the two."""
    length = int(cast("torch.Tensor", args[6])[request].item())
    query_len, group, window = (cast("int", value) for value in args[7:10])
    columns = query_columns(cast("torch.Tensor", args[0]).size(1))
    first_row = length - query_len
    last_row = first_row + (columns - 1) // group
    key_begin = logical_page * PAGE
    key_end = key_begin + PAGE - 1
    inside = key_end <= first_row
    if window >= 0:
        inside = inside and key_begin >= last_row - window
    return inside


def ordered_run(
    args: tuple[object, ...], request: int, shift: int
) -> tuple[int, list[int]]:
    """The first stream step of a request's ordered run and the physical
    pages its steps stream, in order; no pages for a request that is not
    reordered.

    The candidates are the steps behind the head and ahead of the tail, at
    most one sort's capacity. Those whose tile is interior must be one run
    of at least the minimum; its pages are sorted by bucket, and within a
    bucket they keep their source order.
    """
    table = cast("torch.Tensor", args[5])
    reverse = bool(args[14])
    first, last = unit_page_range(args, request)
    count = max(last - first, 0)
    candidates = min(
        max(count - PAGE_ORDER_HEAD - PAGE_ORDER_TAIL, 0), PAGE_ORDER_CAPACITY
    )
    steps = [PAGE_ORDER_HEAD + index for index in range(candidates)]
    inside = [
        interior_tile(args, request, stream_page(first, last, step, reverse))
        for step in steps
    ]
    run = [step for step, keep in zip(steps, inside, strict=True) if keep]
    if len(run) < PAGE_ORDER_MIN_SORTED or run != list(range(run[0], run[-1] + 1)):
        return 0, []
    ids = [
        int(table[request, stream_page(first, last, step, reverse)].item())
        for step in run
    ]
    # Python's sort is stable.
    return run[0], sorted(ids, key=lambda page: page_order_bucket(page, shift))


def bucketed_page_table(
    args: tuple[object, ...], shift: int
) -> tuple[torch.Tensor, list[int]]:
    """A page table whose source order is the bucketed order of ``args``'s
    table, and the requests it reorders: each ordered step's logical page
    holds the physical page the order streams at that step."""
    table = cast("torch.Tensor", args[5])
    host = host_args(args)
    reverse = bool(args[14])
    result = cast("torch.Tensor", host[5]).clone()
    reordered = []
    for request in range(table.size(0)):
        first_step, ids = ordered_run(host, request, shift)
        if not ids:
            continue
        reordered.append(request)
        first, last = unit_page_range(host, request)
        for offset, page in enumerate(ids):
            result[request, stream_page(first, last, first_step + offset, reverse)] = (
                page
            )
    return result.to(table.device), reordered


def mirror_sort(
    ids: Sequence[int], inside: Sequence[bool], shift: int
) -> tuple[int, int, list[int]]:
    """What the sorting warp leaves for candidates ``ids`` of which
    ``inside`` are interior: the run's first candidate, its length (zero
    when nothing is reordered) and the order, computed the way the device
    code does, lane by lane and round by round."""
    candidates = len(ids)
    assert candidates == len(inside) <= PAGE_ORDER_CAPACITY
    rounds = PAGE_ORDER_CAPACITY // 32
    held = [[0] * 32 for _ in range(rounds)]
    kept = [[0] * 32 for _ in range(rounds)]
    for round_index in range(rounds):
        for lane in range(32):
            index = round_index * 32 + lane
            if index < candidates and inside[index]:
                kept[round_index][lane] = 1
                held[round_index][lane] = ids[index]
    # Each lane's count, first and last, then the warp's by reduction.
    own = [
        sum(kept[round_index][lane] for round_index in range(rounds))
        for lane in range(32)
    ]
    indices = [
        [
            round_index * 32 + lane
            for round_index in range(rounds)
            if kept[round_index][lane]
        ]
        for lane in range(32)
    ]
    total = sum(own)
    first = min([PAGE_ORDER_CAPACITY, *(lane[0] for lane in indices if lane)])
    last = max([-1, *(lane[-1] for lane in indices if lane)])
    if last - first + 1 != total or total < PAGE_ORDER_MIN_SORTED:
        return 0, 0, []
    counts = [0] * PAGE_ORDER_BUCKETS
    for round_index in range(rounds):
        for lane in range(32):
            if kept[round_index][lane]:
                counts[page_order_bucket(held[round_index][lane], shift)] += 1
    # Each lane's inclusive prefix by doubling shuffles, less its own count.
    prefix = list(counts)
    for bit in range(5):
        lower = [
            prefix[lane - (1 << bit)] if lane >= 1 << bit else 0 for lane in range(32)
        ]
        prefix = [prefix[lane] + lower[lane] for lane in range(32)]
    counts = [prefix[lane] - counts[lane] for lane in range(32)]
    order = [-1] * total
    for round_index in range(rounds):
        if round_index * 32 >= candidates:
            continue
        words = [0] * PAGE_ORDER_BUCKETS
        buckets = [
            page_order_bucket(held[round_index][lane], shift) for lane in range(32)
        ]
        for lane in range(32):
            if kept[round_index][lane]:
                words[buckets[lane]] |= 1 << lane
        for lane in range(32):
            if kept[round_index][lane]:
                below = words[buckets[lane]] & ((1 << lane) - 1)
                position = counts[buckets[lane]] + below.bit_count()
                assert order[position] == -1
                order[position] = held[round_index][lane]
        for lane in range(32):
            if kept[round_index][lane]:
                word = words[buckets[lane]]
                if not word & ((1 << lane) - 1):
                    counts[buckets[lane]] += word.bit_count()
    assert -1 not in order
    return first, total, order


class _Warp:
    """Thirty-two lanes of device code in lockstep: every warp-level
    primitive is a rendezvous of all lanes, as on the device, and what the
    lanes do to shared memory between two of them is checked for races."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.gate = threading.Barrier(32, action=self._settle)
        self.lock = threading.Lock()
        self.kind: list[object] = [None] * 32
        self.given: list[object] = [None] * 32
        self.result: list[object] = [None] * 32
        self.epoch = 0
        self.arrivals: list[int] = []

    def _settle(self) -> None:
        """All lanes are at one primitive: compute what each gets back."""
        (kind,) = set(self.kind)
        if kind == "ballot":
            ballot = sum(1 << lane for lane in range(32) if self.given[lane])
            self.result = [ballot] * 32
        elif kind == "all":
            self.result = [all(self.given)] * 32
        elif isinstance(kind, tuple) and kind[0] == "up":
            (offset,) = kind[1:]
            self.result = [
                self.given[lane - offset] if lane >= offset else self.given[lane]
                for lane in range(32)
            ]
        elif isinstance(kind, tuple):
            (mask,) = kind[1:]
            self.result = [self.given[lane ^ mask] for lane in range(32)]
        else:
            assert kind == "sync"
        self.epoch += 1

    def _meet(self, kind: object, value: object) -> object:
        lane = self.local.lane
        self.kind[lane] = kind
        self.given[lane] = value
        self.gate.wait(timeout=30)
        return self.result[lane]

    # What the device code calls as ``cute.arch``.
    def vote_ballot_sync(self, predicate: object) -> int:
        return cast("int", self._meet("ballot", bool(predicate)))

    def vote_all_sync(self, predicate: object) -> bool:
        return cast("bool", self._meet("all", bool(predicate)))

    def shuffle_sync_up(self, value: int, offset: int) -> int:
        return cast("int", self._meet(("up", offset), value))

    def shuffle_sync_bfly(self, value: int, mask: int) -> int:
        return cast("int", self._meet(("bfly", mask), value))

    def sync_warp(self) -> None:
        self._meet("sync", None)

    def popc(self, value: int) -> int:
        return (value & 0xFFFFFFFF).bit_count()

    def lanemask_lt(self) -> int:
        return (1 << self.local.lane) - 1

    def atomic_add(self, pointer: _Words, value: int, *, scope: str) -> int:
        assert scope == "cta"
        with self.lock:
            return pointer.update(lambda old: old + value)

    def atomic_or(self, pointer: _Words, value: int, *, scope: str) -> int:
        assert scope == "cta"
        with self.lock:
            return pointer.update(lambda old: old | value)

    def mbarrier_arrive(self, barrier: object) -> None:
        with self.lock:
            self.arrivals.append(self.local.lane)


class _Words:
    """Words of shared memory from an offset: a pointer or a tensor of the
    device code. A plain load or store of a word that another lane stored
    or atomically updated since the last warp-level primitive is a race."""

    def __init__(self, warp: _Warp, size: int, offset: int = 0) -> None:
        self.warp = warp
        self.size = size
        self.offset = offset
        self.values = [0] * size
        # Per word: the epoch and lane of its last plain store, and the
        # epoch of its last atomic update.
        self.stored: list[tuple[int, int]] = [(-1, -1)] * size
        self.updated = [-1] * size

    def __add__(self, index: int) -> _Words:
        other = _Words.__new__(_Words)
        other.__dict__.update(self.__dict__)
        other.offset = self.offset + int(index)
        return other

    def _check(self, index: int, *, store: bool) -> None:
        lane, epoch = self.warp.local.lane, self.warp.epoch
        assert 0 <= index < self.size, (index, self.size)
        stored_epoch, stored_lane = self.stored[index]
        assert stored_epoch != epoch or stored_lane == lane, (
            f"lane {lane} {'stores' if store else 'loads'} word {index} that "
            f"lane {stored_lane} stored since the last warp synchronization"
        )
        assert self.updated[index] != epoch, (
            f"lane {lane} {'stores' if store else 'loads'} word {index} that "
            "was atomically updated since the last warp synchronization"
        )

    def __getitem__(self, index: int) -> int:
        index = self.offset + int(index)
        with self.warp.lock:
            self._check(index, store=False)
            return self.values[index]

    def __setitem__(self, index: int, value: int) -> None:
        index = self.offset + int(index)
        with self.warp.lock:
            self._check(index, store=True)
            self.values[index] = int(value)
            self.stored[index] = (self.warp.epoch, self.warp.local.lane)

    def update(self, change: Callable[[int], int]) -> int:
        """An atomic update of the word at the offset (the warp's lock is
        held): the old value. It may not follow a plain store of the same
        word since the last warp-level primitive."""
        index = self.offset
        assert 0 <= index < self.size
        assert self.stored[index][0] != self.warp.epoch
        old = self.values[index]
        self.values[index] = change(old)
        self.updated[index] = self.warp.epoch
        return old


def _int32(value: object = 0) -> int:
    """A 32-bit signed value, as the device's ``Int32`` holds it."""
    wrapped = int(cast("int", value)) & 0xFFFFFFFF
    return wrapped - (1 << 32) if wrapped >= 1 << 31 else wrapped


def run_device_sort(
    args: tuple[object, ...], request: int, shift: int
) -> tuple[int, int, list[int], int]:
    """Run the device role's own source for one request, 32 lanes in
    lockstep on the host: the run's stream steps ``[first, stop)``, the
    order it left for them, and the page ids it read.

    The role's module is given stand-ins for the DSL: integers for its
    typed values, a rendezvous of all lanes for each warp-level primitive,
    and shared memory that refuses a racing load or store. The interior
    test and the page coordinates are the template's own functions, run as
    plain Python.
    """
    table = cast("torch.Tensor", args[5])
    first, last = unit_page_range(args, request)
    length = int(cast("torch.Tensor", args[6])[request].item())
    query_len, group, window = (cast("int", value) for value in args[7:10])
    columns = query_columns(cast("torch.Tensor", args[0]).size(1))
    reverse = bool(args[14])
    warp = _Warp()
    order = _Words(warp, PAGE_ORDER_CAPACITY)
    buckets = _Words(warp, PAGE_ORDER_BUCKETS)
    members = _Words(warp, PAGE_ORDER_BUCKETS)
    span = _Words(warp, device.SPAN_WORDS)
    row = [int(value) for value in table[request].tolist()]
    reads: list[int] = []

    class Pages:
        def __getitem__(self, index: tuple[int, int]) -> int:
            item, page = index
            assert item == request and 0 <= int(page) < len(row)
            with warp.lock:
                reads.append(int(page))
            return row[int(page)]

    typed = SimpleNamespace(
        Int32=_int32,
        Int64=int,
        min=min,
        max=max,
        const_expr=lambda value: value,
        range_constexpr=range,
    )
    dsl = SimpleNamespace(
        arch=warp,
        make_tensor=lambda pointer, layout: pointer,
        make_layout=lambda shape: shape,
    )
    failures: list[BaseException] = []

    def lane_main(lane: int) -> None:
        warp.local.lane = lane
        try:
            jit_host_body(device.order_unit_pages)(
                Pages(),
                order,
                buckets,
                members,
                span,
                None,
                request,
                first,
                last,
                max(last - first, 0),
                length,
                lane,
                shift,
                columns,
                query_len,
                group,
                window if window >= 0 else None,
                int,
                int if reverse else None,
            )
        except BaseException as error:
            failures.append(error)
            warp.gate.abort()

    with (
        patch.object(device, "cutlass", typed),
        patch.object(device, "cute", dsl),
        patch.object(device, "prims", warp),
        patch.object(
            device,
            "attention_page_coordinates",
            jit_host_body(partial.attention_page_coordinates),
        ),
        patch.object(
            device,
            "logical_attention_tile_is_interior",
            jit_host_body(compute.logical_attention_tile_is_interior),
        ),
        patch.object(
            compute, "_ordered_tile_ranges", jit_host_body(compute._ordered_tile_ranges)
        ),
        patch.object(
            compute,
            "_ordered_window_bounds",
            jit_host_body(compute._ordered_window_bounds),
        ),
    ):
        threads = [
            threading.Thread(target=lane_main, args=(lane,)) for lane in range(32)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    real = [
        error
        for error in failures
        if not isinstance(error, threading.BrokenBarrierError)
    ]
    if real:
        raise real[0]
    assert not failures and sorted(warp.arrivals) == list(range(32))
    run_first, run_stop = span.values
    return run_first, run_stop, order.values[: run_stop - run_first], len(reads)
