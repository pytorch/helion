from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_reference import REFERENCE_ATTENTION_CASES
from helion._testing.cute_paged_reference import check_reference_attention_fixture
from helion._testing.cute_paged_reference import reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,fast_math,lse,scoped,occupancy,pdl,window",
    [
        (*case, window)
        for case, window in zip(
            REFERENCE_ATTENTION_CASES, (31, 2047, 2047, 2047, 31, 31), strict=True
        )
    ],
)
def test_reference_recurrence_and_replay(
    form, query_len, fast_math, lse, scoped, occupancy, pdl, window
):
    # Cover all panels/layouts and both schedules without a Cartesian grid.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    # Window 31 always needs masking; 2047 admits interior and boundary tiles.
    kernel, args = reference_attention_fixture(
        form, query_len, lse, DEVICE, window=window
    )
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        lse,
        scoped,
        occupancy,
        pdl,
        base2_scores=False,
    )
