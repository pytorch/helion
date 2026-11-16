"""A persistent partition matches one unit per CTA, for any lengths, and
every warp of its CTAs finds the units the definitions give it.

Whole units match bit for bit. The balanced partition runs some units in
parts that it merges: those match within the rounding of a reassociated
sum, every other unit bit for bit, and its results do not depend on the
order in which a unit's parts finish."""

from __future__ import annotations

import ctypes
import functools
import operator
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import paged_attention_transport
from helion._compiler.cute import target_facts
from helion._compiler.cute._paged_attention_partition import RECORD_HEADER_BYTES
from helion._compiler.cute._paged_attention_partition import partial_slot_bytes
from helion._compiler.cute._paged_attention_partition import plan_geometry
from helion._compiler.cute._paged_attention_partition import plan_grid_bound
from helion._compiler.cute.paged_attention_partition_bounds import RECORD_COUNTER_BYTES
from helion._compiler.cute.paged_attention_partition_bounds import (
    partition_record_bytes,
)
from helion._testing import DEVICE
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_paged_attention_schedule import paged_attention_reference
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import launch_with_grid
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import partition_attention_kernel
from helion._testing.cute_paged_partition import partition_config
from helion._testing.cute_paged_partition import randomize_partition_inputs
from helion._testing.cute_paged_partition import set_request_lengths
from helion._testing.cute_paged_partition import set_request_pages
from helion._testing.cute_paged_partition_plan import PLAN_WARPS
from helion._testing.cute_paged_partition_plan import applied_skew
from helion._testing.cute_paged_partition_plan import observe_plans
from helion._testing.cute_paged_partition_plan import planner_plans
from helion._testing.cute_paged_partition_plan import planner_units
from helion._testing.cute_paged_partition_plan import reference_partition
from helion._testing.cute_paged_partition_plan import split_units
from helion._testing.poisoned_outputs import poison_
from helion._testing.poisoned_outputs import poisoned_empty
from helion._testing.poisoned_outputs import poisoned_launch
from helion._testing.poisoned_outputs import written_bits
from helion.runtime import cute_cuda_graph
from helion.runtime.cute import paged_partition
from helion.runtime.cute.capture_pdl import WaitMode

if TYPE_CHECKING:
    from collections.abc import Callable

UNITS = 17
CAPACITY = 8


def _sm_count() -> int:
    if not is_cuda():
        pytest.skip("persistent partitions require CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count


def _patterns(requests: int) -> dict[str, list[int]]:
    """Pages per request: what a CTA's units stream decides its barrier phases
    and which query stage each unit takes."""
    generator = torch.Generator().manual_seed(requests)
    pages = [1 + index * 5 % PAGES for index in range(requests)]
    return {
        "uniform": [PAGES] * requests,
        "ragged": torch.randint(
            0, PAGES + 1, (requests,), generator=generator
        ).tolist(),
        # Ranges of two units start at even units: their first, then their
        # later units are the empty ones.
        "even empty": [
            0 if index % 2 == 0 else pages[index] for index in range(requests)
        ],
        "odd empty": [0 if index % 2 else pages[index] for index in range(requests)],
        "all empty": [0] * requests,
        "short": torch.randint(0, 4, (requests,), generator=generator).tolist(),
        # One unit as long as the others together: its CTA runs nothing
        # else, and the CTAs whose share of the work lies inside it idle.
        "one long": [PAGES, *[1] * (requests - 1)],
        "long last": [*[0] * (requests - 1), PAGES],
    }


def _lengths(requests: int) -> dict[str, list[int]]:
    """Request lengths for the short window: a long request starts at a later
    page, whose leading keys a clip leaves out, and ends inside a page."""
    generator = torch.Generator().manual_seed(1000 + requests)
    limit = PAGES * PAGE
    ragged = torch.randint(0, limit + 1, (requests,), generator=generator).tolist()
    return {
        "ragged": ragged,
        "even empty": [
            0 if index % 2 == 0 else ragged[index] for index in range(requests)
        ],
        "odd empty": [0 if index % 2 else ragged[index] for index in range(requests)],
        "all empty": [0] * requests,
        # Shorter than the window: every unit starts at page zero, unclipped.
        "short": torch.randint(
            0, SHORT_WINDOW, (requests,), generator=generator
        ).tolist(),
        # Lengths at and next to page boundaries, in every page of the table.
        "page edges": [
            (index % PAGES + 1) * PAGE - index % 3 for index in range(requests)
        ],
    }


def _launch(
    kernel: Callable[..., object],
    args: tuple[object, ...],
    grid: int | None = None,
    **keywords: object,
) -> list[torch.Tensor]:
    """The bits of one eager call's outputs, every element stored by that call.

    The fixture allocates its outputs with ``torch.empty``, which can return
    the block the reference's result just freed: the outputs start poisoned,
    so a unit no CTA stored cannot pass by keeping the reference's values.
    """
    if grid is None:
        return poisoned_launch(lambda: kernel(*args, **keywords))
    return poisoned_launch(lambda: launch_with_grid(kernel, args, grid))


def _replayed(result: object) -> list[torch.Tensor]:
    """The bits of a captured call's outputs after a replay stored all of them."""
    return [written_bits(value) for value in cast("tuple[torch.Tensor, ...]", result)]


def _assert_same(
    actual: list[torch.Tensor], expected: list[torch.Tensor], what: object
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        assert torch.equal(value, reference), what


@pytest.fixture(scope="module")
def kernels():
    """Seventeen units on a capacity of eight: two or three units per CTA."""
    sms = _sm_count()
    args = partition_attention_args(units=UNITS, device=DEVICE)
    randomize_partition_inputs(args, 0)
    bound = partition_attention_kernel(reserved_sms=sms - CAPACITY).bind(args)
    item = bound.compile_config(partition_config(cute_mma_dag_v_lag=2))
    whole = partition_config("whole", cute_mma_dag_v_lag=2)
    assert f"units={UNITS}, page_limit={PAGES}, q_stages=2" in bound.to_code(whole)
    return args, bound, item, bound.compile_config(whole)


@pytest.fixture(scope="module")
def balanced_kernel(kernels):
    """The balanced partition of the same bind: its CTAs share the pages, so
    most units of the seventeen run in parts on the eight CTAs."""
    _args, bound, _item, _whole = kernels
    config = partition_config("balanced", cute_mma_dag_v_lag=2)
    assert "partition='balanced'" in bound.to_code(config)
    return bound.compile_config(config)


class _DefaultKernels:
    """One bind under the schedule's defaults (guarded capture overlap,
    early retirement without the final join, the leading key clip) and its
    kernels: one unit per CTA and the two persistent partitions, each
    compiled when a test first asks for it. ``window`` None is the
    fixture's own, the whole page table."""

    def __init__(
        self, units: int, reserved_sms: int, seed: int, rows: int, window: int | None
    ) -> None:
        shape = {} if window is None else {"window": window}
        self.args = partition_attention_args(
            units=units, rows=rows, device=DEVICE, **shape
        )
        randomize_partition_inputs(self.args, seed)
        self.bound = partition_attention_kernel(reserved_sms=reserved_sms).bind(
            self.args
        )
        self.rows = rows
        self.short_window = window is not None
        self.knobs = {
            "block_sizes": [32 if rows == 32 else 64],
            "cute_mma_dag_v_lag": 2,
        }

    def _compile(self, partition: str | None):
        config = partition_config(partition, default_policies=True, **self.knobs)
        source = self.bound.to_code(config)
        assert f"query_columns={self.rows}" in source
        if self.short_window:
            assert "key_clip='leading'" in source
        assert "early_tmem_retirement=True" in source
        assert "join_before_exit" not in source
        assert "capture_pdl_wait_mode=capture_pdl_wait_mode" in source
        return self.bound.compile_config(config)

    @functools.cached_property
    def item(self):
        return self._compile(None)

    @functools.cached_property
    def whole(self):
        return self._compile("whole")

    @functools.cached_property
    def balanced(self):
        return self._compile("balanced")


# The default policies' binds by shape: units, reserved SMs, seed, rows and
# window.
_DEFAULT_KERNELS: dict[tuple[int, int, int, int, int | None], _DefaultKernels] = {}


def _default_kernels(
    units: int,
    reserved_sms: int,
    seed: int,
    rows: int = 32,
    window: int | None = SHORT_WINDOW,
) -> _DefaultKernels:
    """The default policies' bind of one shape, once per module: the tests
    of both persistent partitions share its inputs and its one-unit kernel.
    Draft rows and the short window unless named; the shape is the key, so
    a test that names them shares the bind of one that does not."""
    shape = (units, reserved_sms, seed, rows, window)
    if shape not in _DEFAULT_KERNELS:
        _DEFAULT_KERNELS[shape] = _DefaultKernels(*shape)
    return _DEFAULT_KERNELS[shape]


# A schedule case's inputs, bind and one-unit kernel by its knobs, for the
# cases that the tests of both persistent partitions run.
_SCHEDULE_CASES: dict[tuple[tuple[str, object], ...], SimpleNamespace] = {}


def _schedule_case(case: dict[str, object], monkeypatch) -> SimpleNamespace:
    """The bind of one schedule case and its one-unit kernel, compiled once
    per module, with what the case asks of the test: ``knobs`` are the
    config's, the other fields the shape's."""
    sms = _sm_count()
    knobs = dict(case)
    key = tuple(sorted(knobs.items()))
    units = knobs.pop("units", UNITS)
    capacity = knobs.pop("capacity", CAPACITY)
    stages = knobs.pop("q_stages", 2)
    slot_bits = knobs.pop("slot_bits", 32)
    reverse = knobs.pop("reverse_pages", False)
    window = knobs.pop("window", None)
    rows = knobs.pop("rows", 40)
    heads = knobs.pop("heads", 2)
    reference_scores = knobs.pop("reference_scores", False)
    fast_math = knobs.pop("fast_math", True)
    if stages == 1:
        # Between the two shared-memory budgets a persistent CTA keeps one
        # query stage; the kernel then needs less than the device has.
        monkeypatch.setattr(
            target_facts, "per_cta_smem_capacity_bytes", lambda device: 200000
        )
    if slot_bits == 64:
        # A bind this small keeps Int32 slot products; take the wide ones.
        monkeypatch.setattr(
            paged_attention_transport, "partition_slot_bits", lambda *bind: 64
        )
    if rows != 40:
        knobs["block_sizes"] = [rows]
    if key not in _SCHEDULE_CASES:
        args = partition_attention_args(
            units=units,
            rows=rows,
            heads=heads,
            reference_scores=reference_scores,
            reverse_pages=reverse,
            device=DEVICE,
            **({} if window is None else {"window": window}),
        )
        randomize_partition_inputs(args, 1)
        bound = partition_attention_kernel(
            fast_math=fast_math, reserved_sms=sms - capacity, reverse_pages=reverse
        ).bind(args)
        _SCHEDULE_CASES[key] = SimpleNamespace(
            args=args, bound=bound, item=bound.compile_config(partition_config(**knobs))
        )
    shared = _SCHEDULE_CASES[key]
    return SimpleNamespace(
        args=shared.args,
        bound=shared.bound,
        item=shared.item,
        knobs=knobs,
        units=units,
        capacity=capacity,
        stages=stages,
        slot_bits=slot_bits,
        window=window,
        rows=rows,
    )


@pytest.fixture(scope="module", autouse=True)
def _release_shared_kernels():
    """The binds the module's caches hold live as long as the module."""
    yield
    _DEFAULT_KERNELS.clear()
    _SCHEDULE_CASES.clear()


@pytest.fixture(scope="module")
def default_pair():
    """The default policies' bind on the capacity of eight, bound once for
    the module."""
    return _default_kernels(UNITS, _sm_count() - CAPACITY, 5)


@pytest.fixture(scope="module")
def default_kernels(default_pair):
    return default_pair.args, default_pair.item, default_pair.whole


@pytest.fixture(scope="module")
def default_balanced(default_pair):
    """The same bind with its balanced kernel, compiled once."""
    return default_pair.args, default_pair.item, default_pair.balanced


@skipIfNotCUDA()
def test_whole_matches_item_for_every_unit_and_grid(kernels):
    args, _bound, item, whole = kernels
    lengths = cast("torch.Tensor", args[6])
    for name, pages in _patterns(lengths.numel()).items():
        set_request_pages(args, pages)
        expected = _launch(item, args)
        output, lse = expected[0].view(torch.bfloat16), expected[1].view(torch.float32)
        empty = torch.tensor(pages, device=DEVICE).repeat_interleave(2) == 0
        assert torch.isneginf(lse[empty]).all() and not output[empty].any()
        assert torch.isfinite(lse[~empty]).any() or empty.all()
        _assert_same(_launch(whole, args), expected, name)
        # The kernel reads its launch grid: fewer CTAs take more units each,
        # and the CTAs beyond the units have none.
        for grid in (4, 1, UNITS + 3):
            _assert_same(_launch(whole, args, grid), expected, (name, grid))


@skipIfNotCUDA()
def test_whole_matches_item_under_the_default_policies(default_kernels):
    """A unit's first and last page are window edges; the boundary between two
    units of one CTA is not. A clip applied there, or missed at a unit's own
    first page, changes bits against one unit per CTA."""
    args, item, whole = default_kernels
    lengths = cast("torch.Tensor", args[6])
    patterns = _lengths(lengths.numel())
    # The fixture reaches what it is for: units that start past page zero,
    # with leading keys before the window in their first tile.
    first_keys = [length - 4 - SHORT_WINDOW for length in patterns["ragged"]]
    assert sum(key >= PAGE and key % PAGE >= 4 for key in first_keys) >= 4
    for name, values in patterns.items():
        set_request_lengths(args, values)
        expected = _launch(item, args)
        lse = expected[1].view(torch.float32)
        empty = torch.tensor(values, device=DEVICE).repeat_interleave(2) == 0
        assert torch.isneginf(lse[empty]).all()
        assert torch.isfinite(lse[~empty]).any() or empty.all()
        _assert_same(_launch(whole, args), expected, name)
        for grid in (4, 1, UNITS + 3):
            _assert_same(_launch(whole, args, grid), expected, (name, grid))


def _check_overlapped_chain(args, item, tracked, patterns, setter) -> None:
    """Four captured calls, the third fed by the first's pending output."""
    setter(args, next(iter(patterns.values())))
    tracked(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        first = tracked(*args)
        second = tracked(*args)
        dependent = tracked(first[0], *args[1:])
        last = tracked(*args)
    # A call whose inputs no pending write overlaps reads them before it
    # waits for its predecessor; the one fed by a pending output waits first.
    assert modes == [
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
    ], modes
    for name, values in patterns.items():
        setter(args, values)
        expected = _launch(item, args)
        chained = _launch(item, (expected[0].view(torch.bfloat16), *args[1:]))
        # Every replay stores into the outputs of the capture, which hold the
        # previous replay's results: the right values already for each unit
        # whose length two patterns share.
        for result in (first, second, dependent, last):
            poison_(*result)
        graph.replay()
        for result in (first, second, last):
            _assert_same(_replayed(result), expected, name)
        _assert_same(_replayed(dependent), chained, name)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_default_overlap_releases_from_the_tail_of_a_whole_stream(default_kernels):
    """The default policy on the persistent grid: every loader releases the
    dependents four pages before the end of its CTA's stream, whatever the
    stream's length, and captured calls replay changed lengths."""
    args, item, whole = default_kernels
    patterns = _lengths(cast("torch.Tensor", args[6]).numel())
    _check_overlapped_chain(args, item, whole, patterns, set_request_lengths)


def _plan_patterns(units: int, limit: int) -> dict[str, list[int]]:
    """Pages per unit for the plan: what decides where a CTA's share starts."""
    generator = torch.Generator().manual_seed(units + limit)
    ragged = torch.randint(0, limit + 1, (units,), generator=generator).tolist()
    return {
        "uniform": [limit] * units,
        "ragged": ragged,
        "sparse": [
            count if index % 5 == 0 else 0 for index, count in enumerate(ragged)
        ],
        "all empty": [0] * units,
        "one long": [limit, *[0] * (units - 1)],
        "long last": [*[1] * (units - 1), limit],
        # Trip counts a source inside its page table never has: the device
        # clamps them to no page and to the table's extent.
        "out of range": [
            (-1 - index, limit + 1 + index, count)[index % 3]
            for index, count in enumerate(ragged)
        ],
    }


def _check_plans(units: int, limit: int, slot_bits: int, grids: tuple[int, ...]):
    """Every warp of every CTA gets the mirror's plan, which is the units
    the definitions give that CTA."""
    _sm_count()
    observe = observe_plans(units, limit, slot_bits, DEVICE)
    bound = plan_grid_bound(units, limit) if slot_bits == 32 else None
    for name, counts in _plan_patterns(units, limit).items():
        pages = [min(max(count, 0), limit) for count in counts]
        assert (pages != counts) == (name == "out of range")
        for grid in grids:
            observed = observe(counts, grid)
            expected = planner_units(pages, grid, slot_bits, limit)
            for cta, warps in enumerate(observed):
                assert warps == [expected[cta]] * PLAN_WARPS, (name, grid, cta)
            owned = reference_partition(pages, grid, "whole", grid_bound=bound)
            assert [list(range(first, last)) for first, last, _count in expected] == [
                [segment.unit for segment in segments] for segments in owned
            ], (name, grid)


@skipIfNotCUDA()
def test_every_warp_plans_the_units_of_its_cta():
    """Forty units fill one scanning warp and part of a second. The grids
    run from one CTA over fewer and more CTAs than units to more CTAs than
    the units have slots, where the CTAs past the slots idle."""
    assert plan_geometry(40) == (1, 2)
    _check_plans(40, 17, 32, (1, 8, 40, 43, 148, 1000))


# One compile each: Int64 slot products; every warp scanning one unit per
# thread; two and four units per thread, where the plan walks a thread's
# units to the one that holds a slot.
@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize(
    "units,limit,slot_bits",
    [(40, 17, 64), (640, 3, 32), (700, 5, 32), (2000, 2, 64)],
)
def test_every_warp_plans_the_units_of_its_cta_at_scale(units, limit, slot_bits):
    _check_plans(units, limit, slot_bits, (1, 7, 148, 1000))


# Each case is one compile pair. Together they cover every V lag, both query
# stage counts, early and late retirement, the wait policies, four heads per
# request (a lagged V tile keeps its own unit's first head), every probability
# schedule, reversed pages, ranges longer than a warp's lanes, exact math,
# the clips of both sides with lagged tiles under the default policies, and
# Int64 slot products.
SLOW_CASES = {
    "lag0": {"cute_mma_dag_v_lag": 0},
    "wide slots": {"cute_mma_dag_v_lag": 1, "slot_bits": 64},
    "lag1 one stage late": {
        "cute_mma_dag_v_lag": 1,
        "cute_mma_dag_early_tmem_retirement": False,
        "q_stages": 1,
    },
    "lag3 sleep": {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True},
    "four heads lag3": {"heads": 4, "units": 18, "cute_mma_dag_v_lag": 3},
    "four heads lag1 late": {
        "heads": 4,
        "units": 18,
        "cute_mma_dag_v_lag": 1,
        "cute_mma_dag_early_tmem_retirement": False,
    },
    "early": {
        "reference_scores": True,
        "cute_mma_dag_probability_schedule": "early",
        "cute_mma_dag_v_lag": 2,
    },
    "early late": {
        "reference_scores": True,
        "cute_mma_dag_probability_schedule": "early",
        "cute_mma_dag_early_tmem_retirement": False,
        "cute_mma_dag_v_lag": 2,
    },
    "early_tmem one stage": {
        "reference_scores": True,
        "cute_mma_dag_probability_schedule": "early_tmem",
        "cute_mma_dag_v_lag": 1,
        "q_stages": 1,
    },
    "reversed": {"reverse_pages": True, "cute_mma_dag_v_lag": 2},
    "seventy units on two": {"units": 70, "capacity": 2, "fast_math": False},
    "clip both four heads lag3": {
        "heads": 4,
        "units": 18,
        "window": SHORT_WINDOW,
        "default_policies": True,
        "cute_mma_dag_key_clip": "both",
        "cute_mma_dag_v_lag": 3,
    },
    "clip leading reversed lag1": {
        "reverse_pages": True,
        "reference_scores": True,
        "window": SHORT_WINDOW,
        "default_policies": True,
        "cute_mma_dag_probability_schedule": "early",
        "cute_mma_dag_v_lag": 1,
    },
}


@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize("case", list(SLOW_CASES))
def test_whole_matches_item_across_schedule_knobs(case, monkeypatch):
    shared = _schedule_case(SLOW_CASES[case], monkeypatch)
    args, bound, item, knobs = shared.args, shared.bound, shared.item, shared.knobs
    units, window = shared.units, shared.window
    config = partition_config("whole", **knobs)
    source = bound.to_code(config)
    assert f"q_stages={shared.stages}, slot_bits={shared.slot_bits}" in source
    late = knobs.get("cute_mma_dag_early_tmem_retirement") is False
    assert ("early_tmem_retirement=True" in source) != late
    whole = bound.compile_config(config)
    requests = cast("torch.Tensor", args[6]).numel()
    if window is None:
        patterns, setter = _patterns(requests), set_request_pages
    else:
        assert f"key_clip={knobs.get('cute_mma_dag_key_clip', 'leading')!r}" in source
        patterns, setter = _lengths(requests), set_request_lengths
    for name, values in patterns.items():
        setter(args, values)
        expected = _launch(item, args)
        _assert_same(_launch(whole, args), expected, name)
        _assert_same(_launch(whole, args, 1), expected, (name, 1))
        _assert_same(_launch(whole, args, units + 1), expected, (name, "all"))


# The unit counts around the grid boundary of the real device, S SMs: one
# unit, one CTA short of the device, exactly one unit per SM, one CTA with
# two units, two units everywhere, and CTAs with two and with three units.
CAPACITY_UNITS = {
    "one": lambda sms: 1,
    "sms-1": lambda sms: sms - 1,
    "sms": lambda sms: sms,
    "sms+1": lambda sms: sms + 1,
    "2sms": lambda sms: 2 * sms,
    "2sms+4": lambda sms: 2 * sms + 4,
}


@pytest.mark.slow
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("count", list(CAPACITY_UNITS))
def test_whole_matches_item_at_the_device_capacity(count):
    """The generated grid at the real capacity, under the default policies:
    eager, and as captured calls that overlap their predecessors."""
    sms = _sm_count()
    units = CAPACITY_UNITS[count](sms)
    kernels = _default_kernels(units, 0, 2)
    args, item, whole = kernels.args, kernels.item, kernels.whole
    grids = []

    def record(kernel, grid, *arguments, **keywords):
        grids.append(grid)
        return helion.runtime.default_cute_launcher(
            kernel, grid, *arguments, **keywords
        )

    patterns = _lengths(units)
    for name, values in patterns.items():
        set_request_lengths(args, values)
        expected = _launch(item, args)
        _assert_same(_launch(whole, args, _launcher=record), expected, name)
    assert set(grids) == {(min(units, sms),)}
    _check_overlapped_chain(args, item, whole, patterns, set_request_lengths)


@pytest.mark.slow
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_whole_overlaps_captured_calls_with_an_entry_release(kernels):
    args, bound, item, _whole = kernels
    tracked = bound.compile_config(
        partition_config(
            "whole",
            cute_mma_dag_v_lag=2,
            cute_pdl=True,
            cute_pdl_capture_overlap=True,
            cute_pdl_release="auto",
        )
    )
    patterns = _patterns(cast("torch.Tensor", args[6]).numel())
    _check_overlapped_chain(args, item, tracked, patterns, set_request_pages)


@pytest.mark.slow
@skipIfNotCUDA()
def test_whole_matches_item_on_raw_statistics():
    """The raw-statistics epilogue stores each unit's own maximum and denominator."""
    _sm_count()
    kernel, initial = bounded_partial_fixture(4, DEVICE)
    args = list(initial)
    args[7] = 31
    values = tuple(args)
    initialize_paged_attention_inputs(values)
    lengths, first, last = (cast("torch.Tensor", values[i]) for i in (6, 11, 12))
    # The kernel stores its unnormalized output into this argument and
    # allocates the two statistics itself.
    output = [cast("torch.Tensor", values[10])]
    lengths.copy_(lengths.new_tensor([256, 257]))
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=True
    ).bind(values)
    config = partition_config(block_sizes=[32])
    item = bound.compile_config(config)
    config.config["cute_mma_dag_partition"] = "whole"
    source = bound.to_code(config)
    assert "maximum_output=" in source and "partition='whole'" in source
    whole = bound.compile_config(config)
    # Both requests with pages, different ranges, an empty first unit and an
    # empty second one; one CTA then runs both units.
    for begin, end in (
        ([0, 0], [3, 3]),
        ([0, 1], [2, 3]),
        ([2, 0], [1, 3]),
        ([0, 3], [3, 3]),
    ):
        first.copy_(first.new_tensor(begin))
        last.copy_(last.new_tensor(end))
        expected = poisoned_launch(lambda: item(*values), output)
        assert len(expected) == 3
        for grid in (2, 1):
            actual = poisoned_launch(
                lambda grid=grid: launch_with_grid(whole, values, grid), output
            )
            _assert_same(actual, expected, (begin, grid))


# Who takes part in the plan's handoff at barrier 7 changes with the unit
# count, each case one compile pair: seventeen scanning warps and the
# loader's warp, which only waits (544 units); the loader's warp scanning
# too (545: eighteen warps; 640: all twenty); and two units per thread,
# where the loader walks a thread's units, without (641) and with (1,100)
# its own warp among the scanning ones.
HANDOFF_UNITS = {
    544: (1, 17),
    545: (1, 18),
    640: (1, 20),
    641: (2, 11),
    1100: (2, 18),
}


@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize("units", list(HANDOFF_UNITS))
def test_whole_matches_item_at_every_plan_handoff(units):
    """The real kernel's handoff, not the observer's: the scanning warps
    arrive at barrier 7, the loader's warp waits there and plans, and every
    other role reads its plan behind the setup rendezvous. A miscount of the
    barrier's threads hangs the launch or lets the loader plan from tables
    that are not written yet."""
    sms = _sm_count()
    assert plan_geometry(units) == HANDOFF_UNITS[units]
    args = partition_attention_args(units=units, rows=8, device=DEVICE)
    randomize_partition_inputs(args, units)
    bound = partition_attention_kernel().bind(args)
    item = bound.compile_config(partition_config(block_sizes=[8]))
    config = partition_config("whole", block_sizes=[8])
    assert f"units={units}, page_limit={PAGES}" in bound.to_code(config)
    whole = bound.compile_config(config)
    generator = torch.Generator().manual_seed(units)
    patterns = {
        "ragged": torch.randint(0, PAGES + 1, (units,), generator=generator).tolist(),
        "short": torch.randint(0, 3, (units,), generator=generator).tolist(),
        # The last scanning thread's units alone carry work.
        "long last": [*[0] * (units - 1), PAGES],
    }
    for name, pages in patterns.items():
        set_request_pages(args, pages)
        expected = _launch(item, args)
        _assert_same(_launch(whole, args), expected, name)
        _assert_same(_launch(whole, args, 7), expected, (name, "seven CTAs"))
        _assert_same(_launch(whole, args, sms + 5), expected, (name, "idle CTAs"))


@pytest.mark.slow
@skipIfNotCUDA()
def test_whole_walks_hundreds_of_units_in_a_cta():
    """46,341 units on one CTA per SM and on seven CTAs: each scanning thread
    sums 73 units, the plan walks a thread's units to the one that holds a
    slot, and a CTA's roles walk hundreds or thousands of units as one page
    stream. The bind's 834,138 slots keep Int32 products; the Int64 ones are
    the ``wide slots`` case and the plan observer's at scale."""
    _sm_count()
    units = 46341
    assert plan_geometry(units) == (73, 20)
    args = partition_attention_args(units=units, rows=8, device=DEVICE)
    randomize_partition_inputs(args, 3)
    bound = partition_attention_kernel().bind(args)
    item = bound.compile_config(partition_config(block_sizes=[8]))
    whole = bound.compile_config(partition_config("whole", block_sizes=[8]))
    generator = torch.Generator().manual_seed(4)
    set_request_pages(args, torch.randint(0, 3, (units,), generator=generator).tolist())
    expected = _launch(item, args)
    _assert_same(_launch(whole, args), expected, "one CTA per SM")
    _assert_same(_launch(whole, args, 7), expected, "seven CTAs")


def _trips(args: tuple[object, ...]) -> list[int]:
    """The pages each unit streams: the source's trip counts for the lengths."""
    lengths = cast("torch.Tensor", args[6]).tolist()
    query_len, window = cast("int", args[7]), cast("int", args[9])
    heads = cast("torch.Tensor", args[1]).size(3)
    trips = []
    for length in lengths:
        begin = max(length - query_len - window, 0) // PAGE if window >= 0 else 0
        trips += [max((length + PAGE - 1) // PAGE - begin, 0)] * (heads // 2)
    return trips


# How far a unit that ran in parts may be from the same unit in one CTA: a
# fraction of the unit's largest output, and absolutely in its LSE. Measured
# on this fixture (three seeds, six length patterns, four grids, four
# schedules; 729 split units of up to four parts): 0.0058 and 1.9e-6. The
# parts round their probabilities to BF16 against their own maxima, which is
# the rounding a whole unit has against its running maximum, at other points.
SPLIT_OUTPUT = 0.012
SPLIT_LSE = 1e-5
# Both partitions against the FP32 reference, as a fraction of the unit's
# largest output: measured 0.017 for one unit per CTA and 0.033 for a unit in
# parts, whose parts' rounding errors add.
REFERENCE_OUTPUT = 0.05
REFERENCE_LSE = 0.002


def _assert_balanced(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    args: tuple[object, ...],
    grid: int,
    what: object,
    *,
    capacity: int = CAPACITY,
    skew: int = 0,
    reference: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> int:
    """Compare the balanced partition's bits with one unit per CTA's.

    A unit the definitions leave in one CTA has the same bits. A unit in
    parts has pages, so one unit per CTA gives it a largest output above
    zero and a finite LSE, which are asserted. Its outputs are within
    ``SPLIT_OUTPUT`` of that largest output and its LSE within
    ``SPLIT_LSE``; with ``reference``, they are also within
    ``REFERENCE_OUTPUT`` of the largest output and within ``REFERENCE_LSE``
    of the FP32 reference. Those bounds are per unit: the suite's
    per-element tolerance is not checked for a unit in parts. Returns the
    units in parts.
    """
    trips = _trips(args)
    ctas = reference_partition(
        trips, grid, "balanced", workspace_ctas=capacity, skew=skew
    )
    parts = split_units(ctas)
    output, lse = (
        actual[0].view(torch.bfloat16).float(),
        actual[1].view(torch.float32),
    )
    wanted, wanted_lse = (
        expected[0].view(torch.bfloat16).float(),
        expected[1].view(torch.float32),
    )
    for unit in range(len(trips)):
        rows = slice(2 * unit, 2 * unit + 2)
        if unit not in parts:
            assert torch.equal(actual[0][rows], expected[0][rows]), (what, unit)
            assert torch.equal(actual[1][rows], expected[1][rows]), (what, unit)
            continue
        scale = wanted[rows].abs().amax().item()
        assert scale > 0 and torch.isfinite(wanted_lse[rows]).all(), (what, unit)
        moved = (output[rows] - wanted[rows]).abs().amax().item()
        assert moved <= SPLIT_OUTPUT * scale, (what, unit, moved / scale)
        moved = (lse[rows] - wanted_lse[rows]).abs().amax().item()
        assert moved <= SPLIT_LSE, (what, unit, moved)
        if reference is not None:
            exact, exact_lse = reference
            error = (output[rows] - exact[rows].float()).abs().amax().item()
            assert error <= REFERENCE_OUTPUT * scale, (what, unit, error / scale)
            error = (lse[rows] - exact_lse[rows]).abs().amax().item()
            assert error <= REFERENCE_LSE, (what, unit, error)
    return len(parts)


def _counters_are_zero(rows: int = 40) -> bool:
    """Every counter of every workspace this process launched kernels of
    ``rows`` query rows with: the first bytes of each CTA's record."""
    torch.cuda.synchronize()
    record = partition_record_bytes(rows)
    state = paged_partition._WORKSPACES
    buffers = [
        buffer
        for owners in (state.eager, state.captured)
        for key, buffer in owners.items()
        if key[-1][1] == record
    ]
    assert buffers
    return not any(
        buffer.view(-1, record)[:, :RECORD_COUNTER_BYTES].any() for buffer in buffers
    )


@skipIfNotCUDA()
def test_balanced_merges_split_units_within_rounding(kernels, balanced_kernel):
    """Seventeen units on eight CTAs: the CTAs share the pages, so units run
    in two, three and four parts, on grids that split them differently, with
    empty units, CTAs without pages and CTAs past the units."""
    args, _bound, item, _whole = kernels
    lengths = cast("torch.Tensor", args[6])
    split = most = 0
    for name, pages in _patterns(lengths.numel()).items():
        set_request_pages(args, pages)
        assert _trips(args) == pages
        expected = _launch(item, args)
        reference = paged_attention_reference(args, denominator_per_key=True)
        lse = expected[1].view(torch.float32)
        empty = torch.tensor(pages, device=DEVICE).repeat_interleave(2) == 0
        for grid in (None, 4, 3, 1, UNITS + 3):
            actual = _launch(balanced_kernel, args, grid)
            # The CTAs past the workspace's records idle.
            active = min(CAPACITY if grid is None else grid, CAPACITY)
            found = _assert_balanced(
                actual, expected, args, active, (name, grid), reference=reference
            )
            split += found
            most = max(
                most,
                max(
                    map(
                        len,
                        split_units(
                            reference_partition(
                                pages, active, "balanced", workspace_ctas=CAPACITY
                            )
                        ).values(),
                    ),
                    default=0,
                ),
            )
            assert torch.equal(
                torch.isneginf(actual[1].view(torch.float32)),
                empty[:, None].expand_as(lse),
            )
            assert torch.isneginf(lse[empty]).all()
            # The same launch again has the same bits: the fold's order does
            # not follow the order in which the parts finished.
            _assert_same(_launch(balanced_kernel, args, grid), actual, (name, grid))
            assert _counters_are_zero(), (name, grid)
            if grid == 1:
                # One CTA runs every unit whole.
                assert found == 0
                _assert_same(actual, expected, name)
    assert split > 60 and most >= 4


@skipIfNotCUDA()
def test_balanced_keeps_a_units_window_edges_under_the_default_policies(
    default_balanced,
):
    """A part's first and last page are not window edges: a key clip moves
    a tile only at the unit's own first page, wherever the CTAs' ranges cut
    the unit, and the loaders release their dependents from their tails."""
    args, item, balanced = default_balanced
    lengths = cast("torch.Tensor", args[6])
    split = 0
    for name, values in _lengths(lengths.numel()).items():
        set_request_lengths(args, values)
        expected = _launch(item, args)
        for grid in (None, 4, 3, UNITS + 3):
            actual = _launch(balanced, args, grid)
            active = min(CAPACITY if grid is None else grid, CAPACITY)
            split += _assert_balanced(actual, expected, args, active, (name, grid))
            assert _counters_are_zero(32), (name, grid)
    assert split > 20


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_balanced_overlaps_captured_calls_and_keeps_its_counters(default_balanced):
    """Four captured calls share one workspace. A call whose inputs no
    pending write overlaps reads them ahead of its dependency wait, as
    without a workspace: the plan lists the workspace as stored only. Every
    access to it is behind that wait, so the replays of changed lengths, with
    other units in parts, find every counter zero and leave it zero."""
    args, item, balanced = default_balanced
    patterns = _lengths(cast("torch.Tensor", args[6]).numel())
    set_request_lengths(args, patterns["ragged"])
    balanced(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        first = balanced(*args)
        second = balanced(*args)
        dependent = balanced(first[0], *args[1:])
        last = balanced(*args)
    assert modes == [
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
    ], modes
    split = 0
    for _round in range(2):
        for name, values in patterns.items():
            set_request_lengths(args, values)
            expected = _launch(item, args)
            for result in (first, second, dependent, last):
                poison_(*result)
            graph.replay()
            for result in (first, second, last):
                split += _assert_balanced(
                    _replayed(result), expected, args, CAPACITY, name
                )
            # The third call's queries are the first call's outputs, whose
            # split units are the balanced partition's.
            chained_args = (first[0].clone(), *args[1:])
            _assert_balanced(
                _replayed(dependent),
                _launch(item, chained_args),
                chained_args,
                CAPACITY,
                name,
            )
            assert _counters_are_zero(32), name
    assert split > 30


def _check_balanced_plans(
    units: int, limit: int, slot_bits: int, skew: int, workspace_ctas: int, grids
) -> None:
    """Every warp of every CTA gets the mirror's balanced plan: its units,
    pages and slots, its first unit's header, and what a part needs to find
    the other CTAs of its unit."""
    _sm_count()
    observe = observe_plans(
        units,
        limit,
        slot_bits,
        DEVICE,
        partition="balanced",
        skew=skew,
        workspace_ctas=workspace_ctas,
    )
    bound = plan_grid_bound(units, limit) if slot_bits == 32 else None
    for name, counts in _plan_patterns(units, limit).items():
        pages = [min(max(count, 0), limit) for count in counts]
        for grid in grids:
            observed = observe(counts, grid)
            expected = planner_plans(
                pages, grid, slot_bits, limit, "balanced", skew, workspace_ctas
            )
            owned = reference_partition(
                pages,
                grid,
                "balanced",
                grid_bound=bound,
                workspace_ctas=workspace_ctas,
                skew=skew,
            )
            active = expected[0][7]
            for cta, warps in enumerate(observed):
                plan = expected[cta]
                if cta >= active:
                    # An idle CTA's slots and header are not read.
                    assert {warp[:3] for warp in warps} == {plan[:3]}, (name, grid)
                    assert {warp[6:] for warp in warps} == {plan[6:]}, (name, grid)
                    continue
                assert warps == [plan] * PLAN_WARPS, (name, grid, cta)
                assert list(range(plan[0], plan[1])) == [
                    segment.unit for segment in owned[cta]
                ], (name, grid, cta)


@skipIfNotCUDA()
def test_every_warp_plans_the_slots_of_its_balanced_cta():
    """The balanced plan of forty units from one CTA to more CTAs than the
    workspace has records, and with a range skew on a workspace of eight
    records, where the launches of more CTAs idle the rest."""
    _check_balanced_plans(40, 17, 32, 0, 148, (1, 8, 40, 43, 148, 1000))
    _check_balanced_plans(40, 17, 32, 3, 8, (1, 5, 8, 43))


# Each case is one balanced kernel beside the one-unit kernel of its bind,
# which the cases that SLOW_CASES also names share with that test
# (``_schedule_case``), with what the balanced partition adds to
# the knob: no V lag with the serial schedule, whose last PV is not
# outstanding in a part's reduction; three lagged tiles across a part's
# boundary with sleeping waits; four heads per request, where two units
# share a length; the early schedules, the one whose denominators live in
# tensor memory with one query stage; reversed pages, whose parts run the
# reflected steps; late retirement, where every role joins at the end;
# Int64 slot products; two units per scanning thread, whose roles reload
# the lengths; and the clips of both sides under the default policies.
BALANCED_CASES = {
    "lag0 serial": {"cute_mma_dag_v_lag": 0},
    "lag3 sleep": {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True},
    "four heads lag3": {"heads": 4, "units": 18, "cute_mma_dag_v_lag": 3},
    "early": {
        "reference_scores": True,
        "cute_mma_dag_probability_schedule": "early",
        "cute_mma_dag_v_lag": 2,
    },
    "early_tmem one stage late": {
        "reference_scores": True,
        "cute_mma_dag_probability_schedule": "early_tmem",
        "cute_mma_dag_early_tmem_retirement": False,
        "cute_mma_dag_v_lag": 1,
        "q_stages": 1,
    },
    "reversed": {"reverse_pages": True, "cute_mma_dag_v_lag": 2},
    "wide slots": {"cute_mma_dag_v_lag": 1, "slot_bits": 64},
    "two units per thread": {"units": 641, "rows": 8, "capacity": 148},
    "clip both four heads lag3": {
        "heads": 4,
        "units": 18,
        "window": SHORT_WINDOW,
        "default_policies": True,
        "cute_mma_dag_key_clip": "both",
        "cute_mma_dag_v_lag": 3,
    },
}


@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize("case", list(BALANCED_CASES))
def test_balanced_merges_across_schedule_knobs(case, monkeypatch):
    shared = _schedule_case(BALANCED_CASES[case], monkeypatch)
    args, bound, item, knobs = shared.args, shared.bound, shared.item, shared.knobs
    units, capacity, window, rows = (
        shared.units,
        shared.capacity,
        shared.window,
        shared.rows,
    )
    config = partition_config("balanced", **knobs)
    source = bound.to_code(config)
    assert f"q_stages={shared.stages}, slot_bits={shared.slot_bits}" in source
    balanced = bound.compile_config(config)
    requests = cast("torch.Tensor", args[6]).numel()
    if window is None:
        patterns, setter = _patterns(requests), set_request_pages
    else:
        patterns, setter = _lengths(requests), set_request_lengths
    split = 0
    for name, values in patterns.items():
        setter(args, values)
        expected = _launch(item, args)
        for grid in (None, 3, units + 1):
            actual = _launch(balanced, args, grid)
            active = min(capacity if grid is None else grid, capacity, units)
            split += _assert_balanced(
                actual, expected, args, active, (name, grid), capacity=capacity
            )
            _assert_same(_launch(balanced, args, grid), actual, (name, grid))
        assert _counters_are_zero(rows), name
    assert split > 10


# The unit counts around the device's capacity, S SMs, each one balanced
# kernel beside its bind's one-unit kernel on the generated grid under the
# default policies: where the launch has a CTA per unit and every unit runs
# whole (one unit, one CTA short, one per SM), the first split units (one
# more), two units per SM with equal lengths and CTAs with two and three
# units. The rows and the window alternate between the draft and the verify
# width and between a window of a few pages and the whole table, so every
# pairing occurs. The two cases of draft rows on the short window ("one",
# "2sms") have the shape of the whole-unit test's cases of those counts and
# share their bind and one-unit kernel (``_default_kernels``).
BALANCED_CAPACITY = {
    "one": (lambda sms: 1, 32, SHORT_WINDOW),
    "sms-1": (lambda sms: sms - 1, 40, None),
    "sms": (lambda sms: sms, 32, None),
    "sms+1": (lambda sms: sms + 1, 40, SHORT_WINDOW),
    "2sms": (lambda sms: 2 * sms, 32, SHORT_WINDOW),
    "2sms+4": (lambda sms: 2 * sms + 4, 40, None),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("count", list(BALANCED_CAPACITY))
def test_balanced_merges_at_the_device_capacity(count):
    """The generated grid at the real capacity: eager, and as captured calls
    that overlap their predecessors and replay other lengths."""
    sms = _sm_count()
    units, rows, window = BALANCED_CAPACITY[count]
    units = units(sms)
    kernels = _default_kernels(units, 0, 2, rows, window)
    if (rows, window) == (32, SHORT_WINDOW):
        # The whole-unit test's bind and one-unit kernel, as it asks for them.
        assert kernels is _default_kernels(units, 0, 2)
    args, item, balanced = kernels.args, kernels.item, kernels.balanced
    grids = []

    def record(kernel, grid, *arguments, **keywords):
        grids.append(grid)
        return helion.runtime.default_cute_launcher(
            kernel, grid, *arguments, **keywords
        )

    patterns = _lengths(units)
    active = min(units, sms)
    split = 0
    for name, values in patterns.items():
        set_request_lengths(args, values)
        expected = _launch(item, args)
        actual = _launch(balanced, args, _launcher=record)
        split += _assert_balanced(actual, expected, args, active, name, capacity=sms)
    assert set(grids) == {(active,)}
    assert _counters_are_zero(rows)
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        first = balanced(*args)
        second = balanced(*args)
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
    for name, values in patterns.items():
        set_request_lengths(args, values)
        expected = _launch(item, args)
        poison_(*first, *second)
        graph.replay()
        for result in (first, second):
            split += _assert_balanced(
                _replayed(result), expected, args, active, name, capacity=sms
            )
        assert _counters_are_zero(rows), name
    # Ragged lengths split units whenever the CTAs share the pages unevenly
    # among whole units: from two units on.
    assert (split > 0) == (units > 1)


@pytest.mark.slow
@pytest.mark.timeout(240)
@skipIfNotCUDA()
def test_any_finishing_order_of_parts_merges_the_same_bits(kernels, balanced_kernel):
    """Many launches of random lengths on grids of one to eleven CTAs, in
    which the parts of a unit finish in whatever order the device runs
    them: every launch returns, with the bits of its first run and all
    counters zero. Then the same from two host threads on two streams,
    which launch concurrently on one device with a workspace each."""
    args, bound, item, _whole = kernels
    generator = torch.Generator().manual_seed(9)
    lengths = cast("torch.Tensor", args[6])
    cases = []
    for _ in range(40):
        pages = torch.randint(0, PAGES + 1, (lengths.numel(),), generator=generator)
        grid = int(torch.randint(1, 12, (1,), generator=generator))
        cases.append((pages.tolist(), grid))
    seen = {}
    split = 0
    for index, (pages, grid) in enumerate(cases):
        set_request_pages(args, pages)
        expected = _launch(item, args)
        actual = _launch(balanced_kernel, args, grid)
        split += _assert_balanced(
            actual, expected, args, min(grid, CAPACITY), (index, grid)
        )
        seen[index] = actual
    assert split > 100 and _counters_are_zero()
    for _round in range(12):
        for index, (pages, grid) in enumerate(cases):
            set_request_pages(args, pages)
            _assert_same(_launch(balanced_kernel, args, grid), seen[index], index)
    assert _counters_are_zero()

    failures = []

    def stream_of_launches(worker: int, poisoned: list[torch.Tensor]) -> None:
        try:
            # The inputs are built and written on the stream that launches:
            # every launch follows, in its stream's order, the writes of
            # what it reads.
            with torch.cuda.stream(torch.cuda.Stream()):
                own = partition_attention_args(units=UNITS, device=DEVICE)
                randomize_partition_inputs(own, 0)
                for _round in range(8):
                    for index, (pages, grid) in enumerate(cases[worker::2]):
                        set_request_pages(own, pages)
                        result = launch_with_grid(balanced_kernel, own, grid)
                        torch.cuda.current_stream().synchronize()
                        wanted = seen[worker + 2 * index]
                        for value, reference in zip(result, wanted, strict=True):
                            if not any(value is tensor for tensor in poisoned):
                                failures.append((worker, index, "not poisoned"))
                            if not torch.equal(written_bits(value), reference):
                                failures.append((worker, index))
        except Exception as error:
            failures.append((worker, repr(error)))

    # One patch of ``torch.empty`` around both threads: entering and leaving
    # ``poisoned_launch`` in each would interleave two patches of one
    # attribute. Every output is filled on the stream of the thread that
    # allocates it, ahead of that thread's launch, and ``poisoned`` keeps
    # them all, so no launch gets a block an earlier one stored into.
    with poisoned_empty() as poisoned:
        workers = [
            threading.Thread(
                target=stream_of_launches, args=(worker, poisoned), daemon=True
            )
            for worker in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=150)
    assert not any(worker.is_alive() for worker in workers), "a launch hangs"
    assert not failures, failures[:4]
    assert _counters_are_zero()


def _poison_partials(records: torch.Tensor) -> None:
    """Overwrite every partial slot of the workspace ``records``; the
    counters stay zero. A part reads a partial only behind the arrival of
    the part that stored it, so the launches that follow do not care."""
    records[:, RECORD_HEADER_BYTES:].fill_(0xA5)


def _stored_partials(records: torch.Tensor, rows: int) -> torch.Tensor:
    """Which of each CTA's four partial slots the last launch stored: per
    head, the part of a unit that began in a lower CTA, then the CTA's own
    part of the unit it is the lowest CTA of. The lowest part of a unit in
    two parts stores its own only when it does not merge from its first
    look at the unit's counter: the unit is then folded from the workspace,
    by whichever part arrived last."""
    torch.cuda.synchronize()
    panel_bytes, slot_bytes = partial_slot_bytes(rows)
    slots = records[:, RECORD_HEADER_BYTES:].view(records.size(0), 4, slot_bytes)
    return (slots[:, :, :panel_bytes] != 0xA5).any(dim=2).cpu()


def _two_part_units(
    pages: list[int], grid: int, capacity: int
) -> dict[int, tuple[int, int]]:
    """The units a launch runs in exactly two parts: for each, its lowest
    CTA and the first step that CTA runs."""
    ctas = reference_partition(pages, grid, "balanced", workspace_ctas=capacity)
    parts: dict[int, list[tuple[int, int]]] = {}
    for cta, segments in enumerate(ctas):
        for segment in segments:
            if not segment.whole:
                parts.setdefault(segment.unit, []).append((cta, segment.step))
    return {
        unit: max(found, key=operator.itemgetter(1))
        for unit, found in parts.items()
        if len(found) == 2
    }


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
def test_a_unit_merges_the_same_bits_from_its_first_look_and_from_the_workspace(
    monkeypatch,
):
    """A unit in two parts is merged by its lowest part inside that part's
    reduction when the other part has arrived by then, and otherwise from
    the workspace by whichever part arrives last: at the end of its walk by
    the higher part, when the lowest part's CTA finished before the higher
    part's had run it. On a grid that is resident at once the first is what
    happens, so a defect of the second would pass every other test here.

    With a capacity of twice the device's SMs the launch runs in two waves:
    the lowest part of the unit between the last CTA of the first wave and
    the first of the second has finished before the other part starts. A
    grid of that capacity over the device's smallest prime factor is
    resident at once and cuts the units at a subset of the same slots, so
    the units between its CTAs run in the same two parts on both grids.
    Their bits are equal whichever way they were merged, and the partial
    slots show that both ways occurred for one of them."""
    sms = _sm_count()
    factor = next(factor for factor in range(2, sms + 1) if sms % factor == 0)
    waves = 2 * sms
    resident = waves // factor
    assert resident <= sms and waves % resident == 0
    real = helion.runtime.get_num_sm
    monkeypatch.setattr(
        helion.runtime,
        "get_num_sm",
        lambda device, *, reserved_sms=0: (
            waves - reserved_sms if device.type == "cuda" else real(device)
        ),
    )
    rows = 40
    args = partition_attention_args(units=waves, rows=rows, device=DEVICE)
    randomize_partition_inputs(args, 3)
    # The fixture's scores spread over tens of units, so one key, and one
    # part, carries nearly all of a row: the other part's term is then
    # below the sum's rounding, in whatever order it is added. Small scores
    # give both parts of a unit a like share of every row, and a fold in
    # another order other bits.
    cast("torch.Tensor", args[0]).mul_(1 / 64)
    bound = partition_attention_kernel().bind(args)
    item = bound.compile_config(partition_config(cute_mma_dag_v_lag=2))
    balanced = bound.compile_config(partition_config("balanced", cute_mma_dag_v_lag=2))
    grids = []

    def record(kernel, grid, *arguments, **keywords):
        grids.append(grid)
        return helion.runtime.default_cute_launcher(
            kernel, grid, *arguments, **keywords
        )

    records = paged_partition.paged_partition_workspace(
        cast("torch.Tensor", args[0]),
        ctas=waves,
        record_bytes=partition_record_bytes(rows),
    ).view(waves, partition_record_bytes(rows))
    generator = torch.Generator().manual_seed(17)
    # Equal units behind a shorter first one: every CTA has the same work,
    # so a wave ends together, and the ranges cut each unit a little
    # earlier than the one before it. The unit between the waves has the
    # last pages of its lowest part in the first wave's last CTA and at
    # least seven pages in the second wave's first. Then ragged lengths.
    patterns = {
        "seventeen": [7, *[PAGES] * (waves - 1)],
        "thirteen": [5, *[13] * (waves - 1)],
        "nearly whole": [1, *[PAGES] * (waves - 1)],
        "ragged": torch.randint(0, PAGES + 1, (waves,), generator=generator).tolist(),
    }
    paths = set()
    compared = between = 0
    for name, pages in patterns.items():
        set_request_pages(args, pages)
        assert _trips(args) == pages
        expected = _launch(item, args)
        launches = {}
        for grid in (waves, resident):
            _poison_partials(records)
            if grid == waves:
                actual = _launch(balanced, args, _launcher=record)
            else:
                actual = _launch(balanced, args, grid)
            _assert_balanced(actual, expected, args, grid, (name, grid), capacity=waves)
            stored = _stored_partials(records, rows)
            assert _counters_are_zero(rows), (name, grid)
            launches[grid] = (actual, stored, _two_part_units(pages, grid, waves))
        actual, stored, parts = launches[waves]
        narrow, narrow_stored, narrow_parts = launches[resident]
        for unit, (lowest, step) in parts.items():
            # The higher part always publishes.
            assert stored[lowest + 1, :2].all(), (name, unit)
            if narrow_parts.get(unit, (None, None))[1] != step:
                continue
            compared += 1
            pair = slice(2 * unit, 2 * unit + 2)
            for value, other in zip(actual, narrow, strict=True):
                assert torch.equal(value[pair], other[pair]), (name, unit)
            paths.update(
                (
                    bool(stored[lowest, 2 + head]),
                    bool(narrow_stored[narrow_parts[unit][0], 2 + head]),
                )
                for head in range(2)
            )
        # The unit between the waves is one of them: its lowest part ran in
        # the first wave's last CTA, which (unless that CTA outlasted its
        # wave by the other part's whole run) found no arrival and
        # published.
        if name != "ragged":
            assert sms in parts and narrow_parts[sms][1] == parts[sms][1], name
            assert parts[sms][0] == sms - 1
            between += int(stored[sms - 1, 2:].sum())
        # The waves again: the same bits, in whatever order the parts finish.
        _assert_same(_launch(balanced, args), actual, name)
        assert _counters_are_zero(rows), name
    assert set(grids) == {(waves,)}
    assert compared > sms
    # Folded from the workspace in the waves, and by the lowest part's own
    # reduction on the resident grid: of the six groups of the three units
    # between the waves, at least one, and normally all.
    assert between >= 1 and (True, False) in paths


@pytest.mark.slow
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_capture_without_a_spare_zeroes_its_counters_in_the_graph(default_balanced):
    """The first capture after an eager launch adopts that launch's spare
    workspace; a second capture with no eager launch in between allocates
    inside the graph and zeroes the counters there. Both replay right, and
    the eager launch that follows makes the next spare without waiting for
    the stream it launches on."""
    args, item, balanced = default_balanced
    values = _lengths(cast("torch.Tensor", args[6]).numel())["ragged"]
    set_request_lengths(args, values)
    expected = _launch(item, args)
    # An eager launch warms the kernel for the capture proof and leaves a
    # zeroed spare.
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    state = paged_partition._WORKSPACES
    # This kernel's spare: other kernels of the process keep theirs.
    geometry = (CAPACITY, partition_record_bytes(32))
    spares = {key: spare for key, spare in state.spares.items() if key[1] == geometry}
    assert spares
    graphs = []
    for _capture in range(2):
        graph = torch.cuda.CUDAGraph()
        with observed_wait_modes() as modes, torch.cuda.graph(graph):
            first = balanced(*args)
            second = balanced(*args)
        assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
        graphs.append((graph, first, second))
        # The capture took the spare, and nothing replaced it.
        assert not any(key in state.spares for key in spares)
    adopted = list(state.captured.values())
    assert any(buffer is spare for buffer in adopted for spare in spares.values())
    for graph, first, second in graphs * 2:
        poison_(*first, *second)
        graph.replay()
        for result in (first, second):
            _assert_balanced(_replayed(result), expected, args, CAPACITY, "replay")
    assert _counters_are_zero(32)
    # The next eager launch zeroes a new spare on the helper's fill stream.
    # It waits for that fill alone, not for the stream it launches on,
    # behind whatever the caller queued there.
    waited = []
    synchronize = torch.cuda.Stream.synchronize

    def observed(stream):
        waited.append(stream.cuda_stream)
        synchronize(stream)

    with patch.object(torch.cuda.Stream, "synchronize", observed):
        _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    assert len(waited) == 1
    assert waited[0] != torch.cuda.current_stream().cuda_stream
    assert all(key in state.spares and not state.spares[key].any() for key in spares)
    del graphs


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_graphs_workspace_is_released_with_the_graph(default_balanced):
    """A raw ``torch.cuda.graph`` leases its workspace on the graph it
    captures into. The buffer serves every replay; once the graph is reset
    or collected, the next launch that makes a buffer returns it to the
    allocator, so a loop that captures again and again holds the buffers of
    the graphs alive at once, not one per capture. A ``cute_cuda_graph``
    releases its buffer at its reset."""
    args, item, balanced = default_balanced
    set_request_lengths(args, _lengths(cast("torch.Tensor", args[6]).numel())["ragged"])
    expected = _launch(item, args)
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    state = paged_partition._WORKSPACES
    geometry = (CAPACITY, partition_record_bytes(32))
    nbytes = geometry[0] * geometry[1]

    def capture(manager=torch.cuda.graph):
        """One captured call: its graph, its outputs and its buffer's key."""
        known = set(state.captured)
        if manager is torch.cuda.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = balanced(*args)
        else:
            with manager() as graph:
                result = balanced(*args)
        (key,) = set(state.captured) - known
        assert key[-1] == geometry
        return graph, result, key

    def replay(graph, result):
        poison_(*result)
        graph.replay()
        _assert_balanced(_replayed(result), expected, args, CAPACITY, "replay")

    def released(key, seconds=10.0):
        """CUDA ends a lease from a thread of its own, behind the graph's
        last replay: wait for it as a later launch would find it."""
        deadline = time.monotonic() + seconds
        while key in state.captured and time.monotonic() < deadline:
            paged_partition._release_destroyed(state)
            time.sleep(0.001)
        return key not in state.captured

    def held():
        return sum(key[-1] == geometry for key in state.captured)

    # Graphs earlier tests destroyed are not this test's.
    torch.cuda.synchronize()
    paged_partition._release_destroyed(state)
    # A live graph keeps its buffer through replays and looks at its lease.
    graph, result, key = capture()
    assert key in state.leases.values()
    for _replay in range(2):
        replay(graph, result)
        assert paged_partition._release_destroyed(state) == 0
    assert key in state.captured
    # Reset: the lease ends, and the buffer's memory is the allocator's.
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(DEVICE)
    graph.reset()
    assert released(key) and key not in state.leases.values()
    assert allocated - torch.cuda.memory_allocated(DEVICE) >= nbytes
    del graph, result
    # Captured again and again, reset or collected, with an eager launch in
    # between (the next capture adopts its spare) and without one (the
    # capture allocates in its graph). Nothing releases here but the
    # launches themselves, which return what the destroyed graphs used: the
    # lease of the graph destroyed last may end a moment later, never an
    # earlier one's, so two buffers are the most that are held.
    in_use = []
    for attempt in range(18):
        graph, result, key = capture()
        replay(graph, result)
        if attempt % 2:
            graph.reset()
        del graph, result
        torch.cuda.synchronize()
        if attempt % 3 == 0:
            _launch(balanced, args)
        assert held() <= 2, attempt
        in_use.append(torch.cuda.memory_allocated(DEVICE))
    # The memory in use does not grow with the captures: it differs by the
    # spare of an eager launch and by the buffer of the last graph.
    assert max(in_use) - min(in_use) <= 2 * nbytes, in_use
    assert released(key)
    # A cute_cuda_graph owns its buffer: no lease, released at its reset.
    graph, result, key = capture(cute_cuda_graph)
    assert key not in state.leases.values()
    replay(graph, result)
    graph.reset()
    assert key not in state.captured


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_spare_fills_stay_out_of_the_graphs_other_threads_capture(default_balanced):
    """One thread captures graphs on the streams of PyTorch's pool, one
    stream after the other until the pool repeats, and inside every capture
    another thread launches eagerly and zeroes the spare workspace the
    capture has just used up. The spare's fill runs on a stream CUDA
    created for the helper, which is none of the pool's: no graph gets the
    fill as a node (each holds the nodes of a capture made alone), no
    capture is invalidated by the wait for the fill, and every launch and
    replay stores the values one unit per CTA stores.

    The captures use the ``thread_local`` mode, because the other thread
    reads its outputs back inside each capture: in the default ``global``
    mode CUDA refuses a thread its own stream synchronizations beside a
    capture. The spare itself needs no such mode of the capture
    (``test_an_eager_launch_makes_its_spare_beside_another_threads_global_capture``)."""
    args, item, balanced = default_balanced
    set_request_lengths(args, _lengths(cast("torch.Tensor", args[6]).numel())["ragged"])
    expected = _launch(item, args)
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    state = paged_partition._WORKSPACES
    spare = (torch.cuda.current_device(), (CAPACITY, partition_record_bytes(32)))
    driver = paged_partition.cuda_driver()

    def nodes() -> int:
        """The nodes of the graph the current stream captures into."""
        capture = driver.stream_capture(torch.cuda.current_stream().cuda_stream)
        assert capture is not None
        count = ctypes.c_size_t()
        assert not driver.library.cuGraphGetNodes(capture[1], None, ctypes.byref(count))
        return count.value

    def capture(stream: torch.cuda.Stream, between: Callable[[], None]):
        """Two captured calls on ``stream`` with ``between`` between them:
        the graph, the calls' outputs and the graph's nodes."""
        # The first call adopts the spare, so it zeroes nothing in the graph.
        assert spare in state.spares
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            first = balanced(*args)
            assert spare not in state.spares
            between()
            second = balanced(*args)
            count = nodes()
        return graph, (first, second), count

    def check(graph, results, what) -> None:
        for result in results:
            poison_(*result)
        graph.replay()
        for result in results:
            _assert_balanced(_replayed(result), expected, args, CAPACITY, what)

    # A capture made alone, and the eager launch that replaces its spare.
    graph, results, alone = capture(torch.cuda.Stream(DEVICE), lambda: None)
    check(graph, results, "alone")
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    graph.reset()
    del graph, results

    go, done = threading.Event(), threading.Event()
    launched: list[list[torch.Tensor]] = []
    failures: list[BaseException] = []
    stop = False

    def launcher() -> None:
        while True:
            go.wait()
            go.clear()
            if stop:
                return
            try:
                launched.append(_launch(balanced, args))
            except BaseException as error:
                failures.append(error)
            done.set()

    def eager_launch_in_the_other_thread() -> None:
        go.set()
        assert done.wait(120)
        done.clear()
        assert not failures, failures
        # It made the next spare, inside this capture.
        assert spare in state.spares

    thread = threading.Thread(target=launcher)
    thread.start()
    streams: dict[int, torch.cuda.Stream] = {}
    try:
        while True:
            stream = torch.cuda.Stream(DEVICE)
            if stream.cuda_stream in streams:
                # PyTorch hands the pool's streams out in turn: all seen.
                break
            streams[stream.cuda_stream] = stream
            graph, results, count = capture(stream, eager_launch_in_the_other_thread)
            assert count == alone, (len(streams), count, alone)
            check(graph, results, ("replay", len(streams)))
            _assert_balanced(launched.pop(), expected, args, CAPACITY, "eager")
            graph.reset()
            del graph, results
    finally:
        stop = True
        go.set()
        thread.join()
    assert len(streams) > 1 and not launched
    assert state.fill_streams[spare[0]].cuda_stream not in streams
    assert _counters_are_zero(32)


def _eager_launch_read_back_later(
    balanced: Callable[..., object], args: tuple[object, ...]
) -> tuple[torch.Tensor, ...]:
    """One eager call whose outputs start poisoned and are read back by the
    caller once no capture is open: a read back is a synchronization of the
    caller's own, which CUDA refuses beside a capture in the ``global`` mode
    whatever the launch does."""
    with poisoned_empty() as fresh:
        outputs = cast("tuple[torch.Tensor, ...]", balanced(*args))
    assert all(any(output is tensor for tensor in fresh) for output in outputs)
    return outputs


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_an_eager_launch_makes_its_spare_beside_another_threads_global_capture(
    default_balanced,
):
    """A raw ``torch.cuda.graph`` in the default ``global`` mode is open on
    one thread, which has used up the spare, while another thread launches
    eagerly and makes the next one. In a thread's default capture mode CUDA
    refuses it the wait for the spare's fill while any thread captures in
    the ``global`` mode, and the refused call invalidates that capture. The
    launch makes its spare with the captures of other threads ignored
    (``paged_partition._other_threads_captures_ignored``): the spare is
    made, the capture stays valid and replays the values one unit per CTA
    stores, and so does the eager launch."""
    args, item, balanced = default_balanced
    set_request_lengths(args, _lengths(cast("torch.Tensor", args[6]).numel())["ragged"])
    expected = _launch(item, args)
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    state = paged_partition._WORKSPACES
    spare = (torch.cuda.current_device(), (CAPACITY, partition_record_bytes(32)))
    assert spare in state.spares
    go, done = threading.Event(), threading.Event()
    launched: list[tuple[torch.Tensor, ...]] = []
    failures: list[BaseException] = []

    def launcher() -> None:
        assert go.wait(120)
        try:
            launched.append(_eager_launch_read_back_later(balanced, args))
        except BaseException as error:
            failures.append(error)
        done.set()

    thread = threading.Thread(target=launcher)
    thread.start()
    prior = torch.cuda.current_stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            first = balanced(*args)
            # The capture took the spare: the eager launch makes the next.
            assert spare not in state.spares
            go.set()
            assert done.wait(120)
            assert not failures, failures
            assert spare in state.spares
            second = balanced(*args)
    finally:
        go.set()
        thread.join()
        # torch leaves the capture stream current when a capture's end
        # raises, which is what an invalidated capture's does.
        torch.cuda.set_stream(prior)
    for result in (first, second):
        poison_(*result)
    graph.replay()
    for result in (first, second):
        _assert_balanced(_replayed(result), expected, args, CAPACITY, "replay")
    (outputs,) = launched
    _assert_balanced(_replayed(outputs), expected, args, CAPACITY, "eager beside")
    assert _counters_are_zero(32)
    graph.reset()


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_an_eager_launch_makes_no_spare_inside_its_threads_capture_elsewhere(
    default_balanced,
):
    """One thread captures a raw ``torch.cuda.graph`` on one device and,
    inside it, launches eagerly on another, where the capture that took the
    spare is gone. CUDA refuses a thread with a capture of its own the
    spare's zeroing or the wait for its fill, in the ``global`` and in
    the ``thread_local`` capture mode, and the refused call invalidates the
    capture: the launch makes no spare while its thread's current stream on
    any device captures. The launch stores the values one unit per CTA
    stores, the capture stays valid and replays right, and the next eager
    launch outside it makes the spare."""
    args, item, balanced = default_balanced
    launch_device = cast("torch.Tensor", args[0]).device
    assert launch_device.index is not None
    other = (launch_device.index + 1) % torch.cuda.device_count()
    set_request_lengths(args, _lengths(cast("torch.Tensor", args[6]).numel())["ragged"])
    expected = _launch(item, args)
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    state = paged_partition._WORKSPACES
    spare = (launch_device.index, (CAPACITY, partition_record_bytes(32)))
    # The spare a capture on the launch device took.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        taken = balanced(*args)
    assert spare not in state.spares
    # The other device's context, its captured operands and the stream of
    # its capture, ahead of the capture: torch's own capture stream is one
    # of the device that was current when the process first captured.
    with torch.cuda.device(other):
        x = torch.ones(1 << 12, device=f"cuda:{other}")
        torch.cuda.synchronize()
        stream = torch.cuda.Stream(other)
    prior = torch.cuda.current_stream()
    elsewhere = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.device(other), torch.cuda.graph(elsewhere, stream=stream):
            assert torch.cuda.current_device() == other
            assert torch.cuda.is_current_stream_capturing()
            doubled = x * 2
            with torch.cuda.device(launch_device):
                assert not torch.cuda.is_current_stream_capturing()
                outputs = _eager_launch_read_back_later(balanced, args)
                assert spare not in state.spares
            result = doubled + 1
    finally:
        torch.cuda.set_stream(prior)
    _assert_balanced(_replayed(outputs), expected, args, CAPACITY, "eager inside")
    x.fill_(3.0)
    elsewhere.replay()
    torch.cuda.synchronize(other)
    assert bool((result == 7).all())
    # Outside the capture the next eager launch makes the spare.
    _assert_balanced(_launch(balanced, args), expected, args, CAPACITY, "eager")
    assert spare in state.spares
    poison_(*taken)
    graph.replay()
    _assert_balanced(_replayed(taken), expected, args, CAPACITY, "replay")
    assert _counters_are_zero(32)
    graph.reset()
    elsewhere.reset()


# A launch applies a skew of at most a thirty-second of its mean share, so
# the fixture's seventeen pages take one only on few CTAs: on three the
# shares reach 102 slots (a skew of two, the bind's largest), on two 153 (of
# four). Each case is one compile pair.
@pytest.mark.slow
@skipIfNotCUDA()
@pytest.mark.parametrize("capacity,skew", [(3, 2), (2, 4)])
def test_balanced_merges_under_a_range_skew(capacity, skew):
    """Skewed ranges cut the units elsewhere: the first CTA runs the most
    pages and the last the fewest. The device applies the bind's largest
    skew to full units and less to shorter ones, down to none."""
    sms = _sm_count()
    args = partition_attention_args(units=UNITS, device=DEVICE)
    randomize_partition_inputs(args, 0)
    bound = partition_attention_kernel(reserved_sms=sms - capacity).bind(args)
    item = bound.compile_config(partition_config(cute_mma_dag_v_lag=2))
    config = partition_config(
        "balanced", cute_mma_dag_v_lag=2, cute_mma_dag_partition_skew=skew
    )
    assert f"partition_skew={skew}" in bound.to_code(config)
    with pytest.raises(helion.exc.InvalidConfig, match="cute_mma_dag_partition_skew"):
        bound.to_code(
            partition_config(
                "balanced", cute_mma_dag_v_lag=2, cute_mma_dag_partition_skew=2 * skew
            )
        )
    skewed = bound.compile_config(config)
    lengths = cast("torch.Tensor", args[6])
    split = 0
    applied = set()
    for name, pages in _patterns(lengths.numel()).items():
        set_request_pages(args, pages)
        expected = _launch(item, args)
        for grid in (None, 2, UNITS + 3):
            actual = _launch(skewed, args, grid)
            active = min(capacity if grid is None else grid, capacity)
            split += _assert_balanced(
                actual,
                expected,
                args,
                active,
                (name, grid),
                capacity=capacity,
                skew=skew,
            )
            applied.add(applied_skew(sum(pages) + len(pages), active, skew))
            _assert_same(_launch(skewed, args, grid), actual, (name, grid))
        assert _counters_are_zero(), name
    # Whole, in part and not at all.
    assert split > 10 and {0, skew} < applied
