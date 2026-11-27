"""A boundary prefetch changes no output bit, on a grid with idle SMs and on
one of more than a wave, eager and as captured calls that overlap their
predecessors, and the host installs a head start only behind a launch of the
same kernel. Nor does the depth the sibling form takes in a launch without
a head start, behind its entry wait or ahead of its dependency wait.

These tests launch the role; they cannot tell which addresses it requests.
A prefetch of an address outside every allocation does not fault on the
devices this runs on, so what the role reads and requests is pinned by the
structural tests of ``test_cute_paged_attention_prefetch.py``."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import partition_attention_kernel
from helion._testing.cute_paged_partition import partition_config
from helion._testing.cute_paged_partition import randomize_partition_inputs
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_stream import IDLE_SMS
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import check_same_bits
from helion._testing.cute_paged_stream import launched_bits
from helion._testing.cute_paged_stream import one_wave_units
from helion._testing.cute_paged_stream import pair_seed
from helion._testing.cute_paged_stream import replayed_bits
from helion._testing.cute_paged_stream import stream_patterns
from helion._testing.poisoned_outputs import poison_
from helion.runtime.cute import capture_pdl_cuda
from helion.runtime.cute.capture_pdl import WAIT_MODE_HEAD_START_SHIFT
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

KEY = "cute_mma_dag_boundary_prefetch"
COLD = "cute_mma_dag_boundary_prefetch_cold"
QUOTAS = "cute_mma_dag_register_quotas"
# SMs the one-wave grid leaves idle (a grid with a form leaves at most a
# fifth of the device's idle), and units past one wave.
# The first is the shared helpers' ``IDLE_SMS``.
LAST_WAVE = 24
# Compiled pairs by (form, schedule knobs): each kernel is compiled once.
_PAIRS: dict[tuple[object, ...], tuple[object, ...]] = {}


def _sm_count() -> int:
    # The shared grid of one wave is the device's SMs less ``IDLE_SMS``.
    sms = one_wave_units() + IDLE_SMS
    if sms < 5 * IDLE_SMS:
        pytest.skip("the one-wave grid leaves at most a fifth of the SMs idle")
    return sms


def _units(form: str) -> int:
    """A grid with idle SMs, or one whose last wave holds a few units."""
    sms = _sm_count()
    return sms - IDLE_SMS if form == "sibling" else sms + LAST_WAVE


def _pair(
    form: str,
    pages: int = 8,
    *,
    window: int | None = None,
    reverse_pages: bool = False,
    cold: int = 0,
    **knobs: object,
) -> tuple[object, ...]:
    """The grid's arguments and its kernel without and with the prefetch,
    under the schedule's default policies: the guarded capture overlap,
    released from the loader's tail on one wave and at the entry of every
    CTA past it, early retirement and the kernel's default key clip.
    ``cold`` is the prefetching kernel's depth in a launch without a head
    start."""
    key = (form, pages, window, reverse_pages, cold, tuple(sorted(knobs.items())))
    if key not in _PAIRS:
        shape = {} if window is None else {"window": window}
        args = partition_attention_args(
            units=_units(form),
            rows=8,
            reverse_pages=reverse_pages,
            device=DEVICE,
            **shape,
        )
        randomize_partition_inputs(args, pair_seed(key))
        bound = partition_attention_kernel(reverse_pages=reverse_pages).bind(args)
        schedule = {"block_sizes": [8], "cute_mma_dag_v_lag": 2, **knobs}
        off = partition_config(default_policies=True, **schedule)
        prefetching = {KEY: pages, **({COLD: cold} if cold else {})}
        on = partition_config(default_policies=True, **prefetching, **schedule)
        assert "boundary_prefetch" not in bound.to_code(off)
        source = bound.to_code(on)
        assert f"boundary_prefetch={pages}, boundary_prefetch_form={form!r}" in source
        assert (f"boundary_prefetch_cold={cold}" in source) == bool(cold)
        assert ("capture_release_tail=True" in source) == (form == "sibling")
        kernels = (bound.compile_config(off), bound.compile_config(on))
        # The capture guard tracks only launches of a warmed kernel.
        set_request_lengths(args, [PAGE] * _units(form))
        for kernel in kernels:
            kernel(*args)
        torch.cuda.synchronize()
        _PAIRS[key] = (args, *kernels)
    return _PAIRS[key]


def _patterns(units: int, window: int) -> dict[str, list[int]]:
    """Request lengths, the shared streams and two more: a served unit's
    range differs from its server's in every pattern but the uniform ones."""
    patterns = stream_patterns(units, window)
    ragged = patterns["ragged"]
    return {
        **patterns,
        "odd empty": [0 if index % 2 else ragged[index] for index in range(units)],
        # The units that serve another on either grid (its first ``IDLE_SMS``
        # or ``LAST_WAVE``) are empty and the units they serve full: a
        # serving lane is the last thread of its CTA to finish.
        "empty servers": [0] * LAST_WAVE + [PAGES * PAGE] * (units - LAST_WAVE),
    }


# The shared helpers, under the names this module calls them by.
_launch = launched_bits
_replayed = replayed_bits
_assert_same = assert_same_bits


@contextlib.contextmanager
def _installed_arguments() -> Iterator[list[tuple[int, int]]]:
    """The wait-mode argument of each captured launch that got one: its mode
    and its head start."""
    installed: list[tuple[int, int]] = []
    install = capture_pdl_cuda.CudaCaptureDriver.set_wait_mode

    def recorded(
        driver: capture_pdl_cuda.CudaCaptureDriver,
        node: int,
        mode: int,
        parameter_sizes: tuple[int, ...],
    ) -> bool:
        done = install(driver, node, mode, parameter_sizes)
        if done:
            installed.append(
                (
                    mode & ((1 << WAIT_MODE_HEAD_START_SHIFT) - 1),
                    mode >> WAIT_MODE_HEAD_START_SHIFT,
                )
            )
        return done

    with patch.object(capture_pdl_cuda.CudaCaptureDriver, "set_wait_mode", recorded):
        yield installed


def _check_pair(
    form: str, pair: tuple[object, ...], window: int, calls: int = 3
) -> None:
    """Every pattern eager, then as ``calls`` captured back-to-back launches
    replayed over the patterns: the kernel with the prefetch stores the bits
    of the kernel without it."""
    args, off, on = cast("tuple[tuple[object, ...], object, object]", pair)
    units = cast("torch.Tensor", args[6]).numel()
    with _installed_arguments() as installed:
        check_same_bits(args, off, on, _patterns(units, window), calls=calls, what=form)
    # Only the launches that overlap a launch of the same kernel get a head
    # start, and only where the kernel reads one: the SMs that launch left
    # idle. Past one wave the argument is the mode alone.
    head = IDLE_SMS if form == "sibling" else 0
    assert installed == [(int(WaitMode.BEFORE_WRITES), head)] * (calls - 1)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("form", ["sibling", "wave_ahead"])
def test_a_boundary_prefetch_keeps_every_output_bit(form):
    """The whole table as the window: units of zero to seventeen pages, of
    which a request serves the first eight."""
    _check_pair(form, _pair(form), PAGES * PAGE - 1)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_cold_launch_prefetch_keeps_every_output_bit():
    """The sibling form with a depth for cold launches. Every eager call and
    the first captured launch have no head start: their first ``IDLE_SMS``
    CTAs serve the waiting units' first two pages, behind the entry wait.
    The later captured launches overlap their predecessor and serve eight.
    Units of zero to seventeen pages, as for the role without the depth."""
    _check_pair("sibling", _pair("sibling", cold=2), PAGES * PAGE - 1)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_cold_launch_ahead_of_its_dependency_wait_keeps_every_output_bit():
    """A launch of the kernel behind a launch of another kernel is in the
    before-writes mode and has no head start: it is a cold launch whose
    serving CTAs read the served units' lengths and page ids and issue
    their requests ahead of the dependency wait, while the launch ahead
    still stores. Its outputs are the bits of the kernel without the role,
    over lengths that change between replays and a page table that holds
    ids past the pool outside every live range."""
    args, off, on = cast(
        "tuple[tuple[object, ...], Callable[..., object], Callable[..., object]]",
        _pair("sibling", cold=2, window=SHORT_WINDOW),
    )
    table, lengths = (cast("torch.Tensor", args[index]) for index in (5, 6))
    units = lengths.numel()
    query_len, window = cast("int", args[7]), cast("int", args[9])
    clean = table.clone()
    patterns = _patterns(units, SHORT_WINDOW)
    try:
        expected = {}
        for name, values in patterns.items():
            set_request_lengths(args, values)
            expected[name] = _launch(off, args)
        graph = torch.cuda.CUDAGraph()
        with (
            _installed_arguments() as installed,
            observed_wait_modes() as modes,
            torch.cuda.graph(graph),
        ):
            captured = [kernel(*args) for kernel in (off, on, off, on, on)]
        assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * 4, modes
        # The kernel with the role behind the other kernel: the mode and no
        # head start, a cold launch. Behind itself it overlaps.
        assert installed == [(1, 0), (1, 0), (1, 0), (1, IDLE_SMS)]
        poison = torch.tensor(
            [cast("torch.Tensor", args[1]).size(0), 2**31 - 1, -1, -(2**31), 2**30],
            dtype=torch.int32,
            device=DEVICE,
        )
        column = torch.arange(table.size(1), device=DEVICE)
        for name, values in patterns.items():
            set_request_lengths(args, values)
            length = lengths.to(torch.int64)[:, None]
            first = (length - query_len - window).clamp(min=0) // PAGE
            last = (length + PAGE - 1) // PAGE
            live = (column >= first) & (column < last)
            fill = poison[(column + torch.arange(units, device=DEVICE)[:, None]) % 5]
            table.copy_(torch.where(live, clean, fill))
            _assert_same(_launch(on, args), expected[name], (name, "eager"))
            for result in captured:
                poison_(*cast("tuple[torch.Tensor, ...]", result))
            graph.replay()
            torch.cuda.synchronize()
            for index, result in enumerate(captured):
                _assert_same(_replayed(result), expected[name], (name, index))
    finally:
        table.copy_(clean)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("form", ["sibling", "wave_ahead"])
def test_a_poisoned_page_table_outside_live_ranges_does_not_fault(form):
    """The role reads page ids of the served unit's own range only, and forms
    a request only for an id inside the pool: a table that holds ids past
    the pool everywhere else runs as the clean one does.

    A smoke test of those gates, not their negative control: a prefetch of
    a wild address does not fault here, so this passes without them. The CPU
    module pins them in the role's source, the two above in
    ``test_the_device_role_reads_page_ids_in_range_and_requests_pages_in_the_pool``
    and the grid and head-start tests in
    ``test_the_serving_rule_gives_a_waiting_unit_one_server``."""
    args, off, on = cast(
        "tuple[tuple[object, ...], object, object]", _pair(form, window=SHORT_WINDOW)
    )
    table, lengths = (cast("torch.Tensor", args[index]) for index in (5, 6))
    units = lengths.numel()
    query_len, window = cast("int", args[7]), cast("int", args[9])
    clean = table.clone()
    patterns = _patterns(units, SHORT_WINDOW)
    try:
        expected = {}
        for name, values in patterns.items():
            set_request_lengths(args, values)
            expected[name] = _launch(off, args)
        graph = torch.cuda.CUDAGraph()
        with observed_wait_modes() as modes, torch.cuda.graph(graph):
            captured = [cast("Callable[..., object]", on)(*args) for _ in range(3)]
        assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * 2, modes
        # Ids no pool has: just past it, the extremes of the dtype, negative.
        poison = torch.tensor(
            [cast("torch.Tensor", args[1]).size(0), 2**31 - 1, -1, -(2**31), 2**30],
            dtype=torch.int32,
            device=DEVICE,
        )
        column = torch.arange(table.size(1), device=DEVICE)
        for name, values in patterns.items():
            set_request_lengths(args, values)
            length = lengths.to(torch.int64)[:, None]
            first = (length - query_len - window).clamp(min=0) // PAGE
            last = (length + PAGE - 1) // PAGE
            live = (column >= first) & (column < last)
            assert bool(live.any()) == any(values)
            fill = poison[(column + torch.arange(units, device=DEVICE)[:, None]) % 5]
            table.copy_(torch.where(live, clean, fill))
            assert int((~live).sum()) > units
            _assert_same(_launch(on, args), expected[name], (form, name, "eager"))
            for result in captured:
                poison_(*cast("tuple[torch.Tensor, ...]", result))
            graph.replay()
            torch.cuda.synchronize()
            for index, result in enumerate(captured):
                _assert_same(_replayed(result), expected[name], (form, name, index))
    finally:
        table.copy_(clean)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_only_a_launch_behind_the_same_kernel_gets_a_head_start():
    """The CTAs that prefetch are those on SMs the launch ahead left idle,
    which the host can tell only of a launch of the same kernel."""
    args, off, on = cast(
        "tuple[tuple[object, ...], Callable[..., object], Callable[..., object]]",
        _pair("sibling"),
    )
    set_request_lengths(args, _patterns(_units("sibling"), 0)["ragged"])
    expected = _launch(off, args)
    graph = torch.cuda.CUDAGraph()
    with (
        _installed_arguments() as installed,
        observed_wait_modes() as modes,
        torch.cuda.graph(graph),
    ):
        captured = [kernel(*args) for kernel in (on, off, off, on, on, off)]
    assert modes == [WaitMode.EARLY] + [WaitMode.BEFORE_WRITES] * 5, modes
    # The kernel without the role gets the mode alone, behind the other
    # kernel and behind itself; the kernel with it a head start behind
    # itself and none behind the other kernel, whose grid is the same.
    assert installed == [(1, 0), (1, 0), (1, 0), (1, IDLE_SMS), (1, 0)]
    for result in captured:
        poison_(*cast("tuple[torch.Tensor, ...]", result))
    graph.replay()
    torch.cuda.synchronize()
    for index, result in enumerate(captured):
        _assert_same(_replayed(result), expected, index)


# One compile pair each: the V lags (the loader's last request is a V tile
# that lags zero to three tiles behind its page, and the role waits behind
# it), sleeping waits, the page counts a config may name, the key clips
# (a clipped tile's request is the part of its box the loader reads),
# whole tiles, the late retirement with its final join (which waits for
# the role's lanes), pages iterated in reverse, the register
# splits a config may name beside the default (the role's warp keeps the
# control quota; the other roles' code is another kernel to the assembler),
# each on the grid where it matters most and one on the other, and the
# sibling form's depth in a cold launch beside its smallest and its largest
# page count (a cold launch requests fewer pages than a lane resolves).
SCHEDULES = {
    "lag0": ("wave_ahead", 8, {"cute_mma_dag_v_lag": 0}),
    "lag1": ("wave_ahead", 8, {"cute_mma_dag_v_lag": 1}),
    "lag3-sleep": (
        "wave_ahead",
        8,
        {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True},
    ),
    "lag0-sibling": ("sibling", 8, {"cute_mma_dag_v_lag": 0}),
    "pages12": ("wave_ahead", 12, {}),
    "pages4-sibling": ("sibling", 4, {}),
    "pages4-cold-sibling": ("sibling", 4, {"cold": 2}),
    "reversed-cold-sibling": ("sibling", 12, {"reverse_pages": True, "cold": 2}),
    "pages12-sibling": ("sibling", 12, {}),
    "clip-both": ("wave_ahead", 8, {"cute_mma_dag_key_clip": "both"}),
    "clip-off-sibling": ("sibling", 8, {"cute_mma_dag_key_clip": "off"}),
    "joined": ("wave_ahead", 8, {"cute_mma_dag_early_tmem_retirement": False}),
    "reversed": ("wave_ahead", 8, {"reverse_pages": True}),
    "reversed-sibling": ("sibling", 8, {"reverse_pages": True}),
    "split-136-80": ("wave_ahead", 8, {QUOTAS: "136/80/80/48"}),
    "split-136-72-sibling": ("sibling", 8, {QUOTAS: "136/72/72/48"}),
    "split-120-96": ("wave_ahead", 8, {QUOTAS: "120/96/96/48"}),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_a_boundary_prefetch_keeps_every_bit_across_schedule_knobs(schedule):
    """A short window: a long request starts at a later page, whose leading
    keys a clip leaves out, and streams fewer pages than a request serves."""
    form, pages, knobs = SCHEDULES[schedule]
    knobs = dict(knobs)
    reverse_pages = bool(knobs.pop("reverse_pages", False))
    cold = cast("int", knobs.pop("cold", 0))
    pair = _pair(
        form,
        pages,
        window=SHORT_WINDOW,
        reverse_pages=reverse_pages,
        cold=cold,
        **knobs,
    )
    _check_pair(form, pair, SHORT_WINDOW, calls=2)
