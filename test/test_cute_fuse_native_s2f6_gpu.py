"""Exact native output bits over the complete raw-byte/scale domain."""

from __future__ import annotations

import pytest
import torch

from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing.cute_native_s2f6 import decode_domain
from helion._testing.cute_native_s2f6 import decode_kernel
from helion._testing.cute_native_s2f6 import grouped_decode_sum


@skipIfNotCUDA()
def test_native_decode_matches_every_bf16_bit():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native S2F6 fusion targets SM100 and SM103")
    # One configuration covers the full 256-by-256 value domain, including
    # negative values, signed zero, subnormals, saturation and canonical NaNs.
    raw, scale, expected = decode_domain()
    code, actual = code_and_output(
        decode_kernel, (raw.cuda(), scale.cuda()), block_sizes=[256]
    )
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in code
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))


@skipIfNotCUDA()
def test_grouped_decode_inside_rolled_reduction():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("native S2F6 fusion targets SM100 and SM103")
    # Broadcast row scales, a non-full row tile, and a partial reduction tile
    # exercise the fused instruction after reduction rolling. All finite terms
    # and partial sums are exactly representable, so reduction order is neutral.
    raw = (torch.arange(3 * 257) % 256).to(torch.uint8).view(torch.int8).reshape(3, 257)
    scale = torch.tensor([[120], [127], [132]], dtype=torch.uint8)
    expected = torch.ldexp(raw.double(), scale.int() - 133).bfloat16().float().sum(-1)
    code, actual = code_and_output(
        grouped_decode_sum,
        (raw.cuda(), scale.cuda()),
        block_sizes=[4],
        num_threads=[4, 32],
        reduction_loops=[64],
    )
    assert "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2" in code
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
