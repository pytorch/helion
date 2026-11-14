from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion._testing.cute_paged_reference import REFERENCE_ATTENTION_CASES
from helion._testing.cute_paged_reference import check_reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,fast_math,lse,scoped,occupancy,pdl",
    REFERENCE_ATTENTION_CASES,
)
def test_base2_recurrence_and_replay(
    form, query_len, fast_math, lse, scoped, occupancy, pdl
):
    # Cover all panels/layouts and both schedules without a Cartesian grid.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = base2_attention_fixture(form, query_len, lse, DEVICE)
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        lse,
        "attention_scoped" if scoped else "attention",
        occupancy,
        pdl,
        base2_scores=True,
    )
