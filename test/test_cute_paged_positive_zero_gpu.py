from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_decode_reference import check_full_domain_decode
from helion._testing.cute_paged_positive_zero import POSITIVE_ZERO_ATTENTION_CASES
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_reference import check_reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "post_scale,stride,swizzle",
    [
        (-22, 4, False),
        (-14, 4, False),
        (-13, 4, False),
        (-6, 4, False),
        (0, 4, False),
        (-22, 8, False),
        (-22, 4, True),
    ],
)
def test_positive_zero_decode_bits_and_changed_replay(post_scale, stride, swizzle):
    # The exponent endpoints and zero-domain boundary share the packed layout;
    # padded and swizzled scales force each generic path without a product grid.
    # Each configuration covers all 65,536 byte/scale pairs, signed zeros,
    # NaNs and infinities, plus sparse boundary and changed-data graph replays.
    check_full_domain_decode(
        128, 32, stride, swizzle, True, post_scale, positive_zero=True
    )


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,schedule,occupancy,query_len,post_scale,fast_math",
    POSITIVE_ZERO_ATTENTION_CASES,
)
def test_positive_zero_attention_source_and_replay(
    form, schedule, occupancy, query_len, post_scale, fast_math
):
    # Four source schedules cover reduced and per-key denominators, N8/N32/N40,
    # both math policies, and the paired base2/reflected source form.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    reference = form == "mapped"
    kernel, args = positive_zero_attention_fixture(
        query_len,
        post_scale,
        DEVICE,
        form=form,
        reference_scores=reference,
    )
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        query_len,
        fast_math,
        True,
        schedule,
        occupancy,
        False,
        base2_scores=reference,
        reference_scores=reference,
        reverse_pages=reference,
        output_reciprocal=True,
        key_post_scale_log2=-6,
        value_post_scale_log2=post_scale,
        output_scale_log2=-post_scale,
        value_positive_zero=True,
    )
