"""A raw V ring counted in tiles changes no output bit: five tiles on a
persistent partition (``cute_mma_dag_value_ring_tiles``), eager and as
captured calls that overlap their predecessors, over units of unequal
length that the balanced partition splits between CTAs.

Each launch is also the test of the shared-memory accounting of the ring:
CUDA rejects a launch whose table the device does not hold."""

from __future__ import annotations

import pytest

from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_stream import PERSISTENT_CTAS
from helion._testing.cute_paged_stream import check_persistent_bits
from helion._testing.cute_paged_stream import device_shared_memory

TILES = "cute_mma_dag_value_ring_tiles"
PARTITION = "cute_mma_dag_partition"
ORDER = "cute_mma_dag_page_order"
LEAD = "cute_mma_dag_clipped_page_lead"
V_LAG = "cute_mma_dag_v_lag"
BALANCED = {PARTITION: "balanced"}
# The walk a search pairs the ring with.
WALK = {**BALANCED, ORDER: "reversed", LEAD: 8}


def _check(
    rows: int = 8, calls: int = 3, window: int | None = None, **schedule: object
) -> None:
    """The schedule's kernel with five tiles stores the bits of the kernel
    with the ring in pages, and renders the count on the capacity it was
    admitted on."""
    rendered = {
        "value_ring_tiles": "5",
        "shared_memory_capacity": str(device_shared_memory()),
    }
    options = {} if window is None else {"window": window}
    check_persistent_bits(
        {TILES: 5},
        PERSISTENT_CTAS,
        rendered,
        rows=rows,
        calls=calls,
        **options,
        **schedule,
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("v_lag", [0, 2])
def test_a_ring_of_five_tiles_keeps_every_output_bit(v_lag):
    """The ring's five slots are taken by a stream of up to thirty-four
    tiles a unit, across unit boundaries; V is issued with its K and two
    tiles behind it."""
    _check(**BALANCED, **{V_LAG: v_lag})


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_the_widest_persistent_table_launches_with_five_tiles():
    """Forty query columns with two query stages: the table a verify-shaped
    kernel has, which holds five tiles and not a third page per head."""
    _check(rows=40, calls=2, **BALANCED)


# One compile pair each: the other V lags under the ring, whole units, the
# reversed walk a search pairs the ring with, and a short window (a unit
# streams at most four pages, fewer than the ring holds, and its first tile
# is clipped).
SCHEDULES = {
    "lag1": ({**BALANCED, V_LAG: 1}, {}),
    "lag3-sleep": ({**BALANCED, V_LAG: 3, "cute_mma_dag_sleep_waits": True}, {}),
    "whole": ({PARTITION: "whole"}, {}),
    "walk": (WALK, {}),
    "short-window": (BALANCED, {"window": SHORT_WINDOW}),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_five_tiles_keep_every_bit_across_schedule_knobs(schedule):
    knobs, options = SCHEDULES[schedule]
    _check(calls=2, **options, **knobs)
