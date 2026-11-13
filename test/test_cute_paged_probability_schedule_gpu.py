from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_probability import check_probability_schedule_replay


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "schedule,form,query_len,base2,reverse,fast_math,sleep,lag,return_lse,"
    "capture_overlap",
    [
        ("early", "mapped", 1, False, False, False, False, 0, True, False),
        ("early", "per_key", 3, True, True, True, True, 3, False, False),
        ("early", "mapped", 5, True, True, False, False, 1, True, True),
        ("early_tmem", "per_key", 1, True, True, False, True, 2, True, False),
        ("early_tmem", "mapped", 4, False, False, True, False, 0, False, False),
        ("early_tmem", "per_key", 5, False, True, True, True, 3, True, True),
    ],
)
def test_probability_schedule_source_and_graph_replay(
    schedule,
    form,
    query_len,
    base2,
    reverse,
    fast_math,
    sleep,
    lag,
    return_lse,
    capture_overlap,
):
    # Each schedule covers all three TMEM widths. Layout, math, direction,
    # waiting, V lag and optional LSE are paired instead of multiplied.
    # N=32 includes a ragged 24-row query; both N=40 cases exercise the seed's
    # query-TMA and capture-overlap combination without adding compilations.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_probability_schedule_replay(
        schedule,
        form,
        query_len,
        base2,
        reverse,
        fast_math,
        sleep,
        lag,
        return_lse,
        DEVICE,
        capture_overlap=capture_overlap,
    )
