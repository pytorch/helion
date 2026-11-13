from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_reciprocal import RECIPROCAL_ATTENTION_CASES
from helion._testing.cute_paged_reciprocal import reciprocal_attention_fixture
from helion._testing.cute_paged_reference import check_reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,reference,base2,reverse,schedule,occupancy,fast_math,lse,pdl",
    RECIPROCAL_ATTENTION_CASES,
)
def test_reciprocal_source_and_changed_input_replay(
    form, query_len, reference, base2, reverse, schedule, occupancy, fast_math, lse, pdl
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = reciprocal_attention_fixture(
        form,
        query_len,
        lse,
        DEVICE,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
    )
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        lse,
        schedule,
        occupancy,
        pdl,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
        output_reciprocal=True,
    )
