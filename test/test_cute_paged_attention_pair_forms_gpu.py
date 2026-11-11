from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion._testing.cute_paged_reference import check_reference_attention_fixture
from helion._testing.cute_paged_reference import reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,fast_math,lse,base2,reverse,index_dtype",
    [
        ("mapped", 1, False, True, True, False, torch.int32),
        ("per_key", 5, True, False, True, False, torch.int64),
        ("mapped", 4, True, False, False, True, torch.int64),
        ("per_key", 1, False, True, False, True, torch.int32),
        ("mapped", 5, False, True, True, True, torch.int64),
        ("per_key", 4, True, True, True, True, torch.int32),
    ],
)
def test_paired_explicit_source_forms(
    form, query_len, fast_math, lse, base2, reverse, index_dtype
):
    # Six configurations cover each new semantic form, physical panel, I/O
    # layout and math setting; reference/replay data changes are shared.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    fixture = base2_attention_fixture if base2 else reference_attention_fixture
    kernel, args = fixture(form, query_len, lse, DEVICE)
    args = (*args, reverse)
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        lse,
        "attention_paired",
        1,
        False,
        base2_scores=base2,
        reverse_pages=reverse,
        index_dtype=index_dtype,
    )
