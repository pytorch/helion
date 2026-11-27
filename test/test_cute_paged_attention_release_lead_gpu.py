"""A lead of the paired loader's tail release changes no output bit: eager
and as captured calls that overlap their predecessors, for streams longer
and shorter than the lead, with one unit per CTA and on the persistent
partitions.

These tests launch the release; they cannot tell when it happens. The step
is pinned by ``test_cute_paged_attention_release_lead.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import check_same_bits
from helion._testing.cute_paged_stream import stream_pair
from helion._testing.cute_paged_stream import stream_patterns

if TYPE_CHECKING:
    import torch

KEY = "cute_mma_dag_release_lead"


def _check(
    lead: int,
    window: int,
    calls: int,
    *,
    reverse_pages: bool = False,
    **schedule: object,
) -> None:
    """The kernel with the release's own lead against the kernel with
    ``lead``, over streams of every length the table has."""
    args, default, longer, plain, source = stream_pair(
        {KEY: lead}, schedule=schedule, window=window, reverse_pages=reverse_pages
    )
    rendered = body_keywords(plain)
    assert rendered["capture_release_tail"] == "True"
    assert "capture_release_lead" not in rendered
    rendered = body_keywords(source)
    assert rendered["capture_release_tail"] == "True"
    assert rendered["capture_release_lead"] == str(lead)
    units = cast("torch.Tensor", args[6]).numel()
    check_same_bits(
        args,
        default,
        longer,
        stream_patterns(units, window),
        calls=calls,
        what=(lead, schedule),
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("lead", [8, 16])
def test_a_lead_keeps_every_output_bit(lead):
    """The whole table as the window: units of zero to seventeen pages, so
    most streams are shorter than the lead and release at their first page,
    and the longest release one or nine pages into theirs."""
    _check(lead, PAGES * PAGE - 1, 3)


# One compile pair each: the V lags (a lagged V tile is requested behind the
# release), sleeping waits, a clipped first tile under each clip, the late
# retirement with its final join, pages iterated in reverse, and the
# persistent partitions, whose lead is that of a CTA's whole stream of
# several units.
SCHEDULES = {
    "lag0": (16, {"cute_mma_dag_v_lag": 0}),
    "lag3-sleep": (8, {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True}),
    "clip-both": (16, {"cute_mma_dag_key_clip": "both"}),
    "joined": (8, {"cute_mma_dag_early_tmem_retirement": False}),
    "reversed": (16, {"reverse_pages": True}),
    "whole": (8, {"cute_mma_dag_partition": "whole"}),
    "balanced": (16, {"cute_mma_dag_partition": "balanced"}),
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_a_lead_keeps_every_bit_across_schedule_knobs(schedule):
    """A short window: a long request starts at a later page, whose leading
    keys a clip leaves out, and streams at most four pages."""
    lead, knobs = SCHEDULES[schedule]
    knobs = dict(knobs)
    reverse_pages = bool(knobs.pop("reverse_pages", False))
    _check(lead, SHORT_WINDOW, 2, reverse_pages=reverse_pages, **knobs)
