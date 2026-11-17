from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_paged_decode_reference import check_full_domain_decode
from helion._testing.cute_paged_postscale import POSTSCALE_ATTENTION_CASES
from helion._testing.cute_paged_postscale import postscale_attention_fixture
from helion._testing.cute_paged_reference import check_reference_attention_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "exponent,transpose,stride,swizzle",
    [
        (-22, False, 4, False),
        (-22, True, 4, False),
        (-7, False, 8, False),
        (-22, True, 4, True),
        (-18, True, 8, False),
        (-14, False, 4, True),
        (-7, True, 4, False),
        (-8, True, 4, False),
        (-3, False, 4, False),
        (-18, True, 4, False),
        (-14, True, 4, False),
        (-6, True, 4, False),
    ],
)
def test_postscaled_decode_full_domain_and_replay(exponent, transpose, stride, swizzle):
    # Cover underflow, intermediate gap, and original fused domains with both
    # decoders and all layout routes, without a Cartesian compile grid. Every
    # chosen exponent checks all 65,536 byte pairs and mixed boundary replays.
    # Packed -7/-8 cover both one-bit and multi-bit subnormal rounding gaps.
    # Packed -18/-14 cases exercise an inner and endpoint zero exponent;
    # generic layouts cannot take the complete-tile arm. The -6 control pins
    # the nonapplicable path without the p<-6 clamp and packed multiply.
    check_full_domain_decode(128, 32, stride, swizzle, transpose, exponent)


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,rows,key,value,reference,base2,reverse,schedule,occupancy,fast,lse",
    POSTSCALE_ATTENTION_CASES,
)
def test_postscaled_attention_source_and_replay(
    form, rows, key, value, reference, base2, reverse, schedule, occupancy, fast, lse
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = postscale_attention_fixture(
        form,
        rows,
        lse,
        DEVICE,
        key=key,
        value=value,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
    )
    check_reference_attention_fixture(
        kernel,
        args,
        form,
        rows,
        fast,
        lse,
        schedule,
        occupancy,
        False,
        reference_scores=reference,
        base2_scores=base2,
        reverse_pages=reverse,
        output_reciprocal=True,
        key_post_scale_log2=key,
        value_post_scale_log2=value,
        output_scale_log2=-value,
    )
