from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_v_lag import check_paired_v_lag_extreme_bounds
from helion._testing.cute_paged_v_lag import check_paired_v_lag_replay


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "lag,form,query_len,base2,reverse,fast_math,heads",
    [
        (0, "mapped", 1, False, False, False, 2),
        (1, "per_key", 4, False, False, True, 4),
        (2, "mapped", 5, False, False, False, 2),
        (3, "per_key", 1, False, False, True, 2),
        (0, "per_key", 5, True, True, True, 2),
        (1, "mapped", 1, True, True, False, 2),
        (2, "per_key", 4, True, True, True, 4),
        (3, "mapped", 5, True, True, False, 2),
    ],
)
def test_paired_v_lag_and_source_replay(
    lag, form, query_len, base2, reverse, fast_math, heads
):
    # Each lag runs ordinary forward and explicit base2/reverse sources. The
    # remaining panel/layout/math axes are paired rather than multiplied.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_paired_v_lag_replay(
        lag, form, query_len, base2, reverse, fast_math, DEVICE, heads=heads
    )


@skipIfNotCUDA()
@pytest.mark.parametrize("lag", [0, 1, 2, 3])
def test_paired_v_lag_empty_int64_bounds(lag):
    # Each compile-time drain length needs its own empty-to-nonempty check.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_paired_v_lag_extreme_bounds(lag, DEVICE)


@skipIfNotCUDA()
def test_serial_nonreference_retirement_replay():
    # Match the seeded serial/non-PDL schedule at the missing N32 panel width.
    # The ordinary per-key body exercises its in-loop PV drain, full O/LSE,
    # changed lengths/data and backing-buffer graph replay checks.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    check_paired_v_lag_replay(
        3,
        "per_key",
        4,
        False,
        False,
        False,
        DEVICE,
        early_tmem_retirement=True,
    )
