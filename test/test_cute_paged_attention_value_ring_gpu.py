"""A raw V ring of three pages per head changes no output bit: eager and as
captured calls that overlap their predecessors, under every V lag, beside
the other stream knobs, with one unit per CTA and on the persistent
partitions.

Each launch is also the test of the shared-memory accounting: the config
contract admits a depth where the body's table fits the device, and CUDA
rejects a launch whose table does not. Layouts close to the device's
capacity are among these: 222,040 of a B200's 232,448 bytes, where the
largest the contract admits there takes 226,136."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute.paged_attention_shared_memory import (
    paired_shared_memory_bytes,
)
from helion._compiler.cute.paged_attention_shared_memory import value_ring_reason
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_stream import IDLE_SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import check_same_bits
from helion._testing.cute_paged_stream import device_shared_memory
from helion._testing.cute_paged_stream import one_wave_units
from helion._testing.cute_paged_stream import stream_pair
from helion._testing.cute_paged_stream import stream_patterns

if TYPE_CHECKING:
    import torch

KEY = "cute_mma_dag_value_ring"
LEAD = "cute_mma_dag_release_lead"
V_LAG = "cute_mma_dag_v_lag"
PARTITION = "cute_mma_dag_partition"


def _check(
    window: int,
    calls: int,
    *,
    rows: int = 8,
    reverse_pages: bool = False,
    reserved_sms: int = 0,
    candidate: dict[str, object] | None = None,
    **schedule: object,
) -> None:
    """The kernel with two pages of raw V per head against the kernel with
    three (and ``candidate``'s other knobs), over streams of every length
    the table has."""
    args, default, deeper, plain, source = stream_pair(
        {KEY: 3, **(candidate or {})},
        schedule=schedule,
        window=window,
        reverse_pages=reverse_pages,
        rows=rows,
        reserved_sms=reserved_sms,
    )
    assert "value_ring" not in plain
    rendered = body_keywords(source)
    assert rendered["value_ring"] == "3"
    assert rendered["shared_memory_capacity"] == str(device_shared_memory())
    units = cast("torch.Tensor", args[6]).numel()
    check_same_bits(
        args,
        default,
        deeper,
        stream_patterns(units, window),
        calls=calls,
        what=(rows, schedule, candidate),
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("v_lag", [0, 2])
def test_a_deeper_ring_keeps_every_output_bit(v_lag):
    """The whole table as the window: units of zero to seventeen pages, so
    the ring's three slots per head are taken up to six times each and
    streams end at every slot. V is issued with its K, the lag the seeds
    pair the depth with, and two tiles behind it, the default config's."""
    _check(PAGES * PAGE - 1, 3, **{V_LAG: v_lag})


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_widest_one_unit_table_launches():
    """Forty query columns, the widest panel: the body's table takes 213,008
    bytes with the query map. The config contract replays the table with
    every optional part such a body can have and admits the depth from
    214,936 bytes (``value_ring_reason``); a device that grants less cannot
    bind this config."""
    assert paired_shared_memory_bytes(40, value_ring=3, query_map=True) == 213008
    if value_ring_reason(3, device_shared_memory(), 40) is not None:
        pytest.skip(
            "the device grants a CTA less than the contract admits the depth on"
        )
    _check(PAGES * PAGE - 1, 2, rows=40, **{V_LAG: 0})


# One compile pair each: the other V lags (a lagged V tile takes the slot of
# its own page's step), sleeping waits, the seeds' pairing with the longer
# lead of the tail release, a boundary prefetch on the idle control warp, a
# clipped first tile, the late retirement with its final join, pages
# iterated in reverse, and the persistent partitions at 32 query columns
# with several units per CTA: with two query stages their table takes
# 221,880 bytes (``whole``) and 222,040 (``balanced``) at a B200's grid of
# 128 units, of the 232,448 the device grants (the largest the contract
# admits there is 226,136, ``balanced`` at 640 units), and their stream
# crosses unit boundaries inside the ring.
SCHEDULES = {
    "lag1": ({V_LAG: 1}, {}),
    "lag3-sleep": ({V_LAG: 3, "cute_mma_dag_sleep_waits": True}, {}),
    "lead16": ({V_LAG: 0}, {"candidate": {LEAD: 16}}),
    "prefetch8": ({V_LAG: 0, "cute_mma_dag_boundary_prefetch": 8}, {}),
    "clip-both": ({"cute_mma_dag_key_clip": "both"}, {}),
    "joined": ({"cute_mma_dag_early_tmem_retirement": False}, {}),
    "reversed": ({V_LAG: 0}, {"reverse_pages": True}),
    "whole-32": ({PARTITION: "whole", V_LAG: 0}, {"rows": 32, "persistent": True}),
    "balanced-32": (
        {PARTITION: "balanced"},
        {"rows": 32, "persistent": True},
    ),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_a_deeper_ring_keeps_every_bit_across_schedule_knobs(schedule):
    """A short window: a long request starts at a later page, whose leading
    keys a clip leaves out, and streams at most four pages, so a unit ends
    inside the ring's first turn."""
    knobs, options = SCHEDULES[schedule]
    options = dict(options)
    reserved_sms = 0
    if options.pop("persistent", False):
        # A third of the grid's units per SM left to the partition: every
        # CTA runs three units or parts of them. (The grid's units and the
        # SMs it leaves idle are the device's SMs.)
        units = one_wave_units()
        reserved_sms = units + IDLE_SMS - units // 3
        needed = paired_shared_memory_bytes(
            32,
            value_ring=3,
            query_stages=2,
            units=units,
            balanced=knobs[PARTITION] == "balanced",
            scanned_bounds=True,
        )
        if device_shared_memory() < needed:
            pytest.skip("the device grants a CTA less than the table needs")
    _check(SHORT_WINDOW, 2, reserved_sms=reserved_sms, **options, **knobs)
