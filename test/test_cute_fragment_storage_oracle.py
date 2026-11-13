"""Portable equivalence of the bounded oracle's FP32-input FMA emulation."""

from __future__ import annotations

from fractions import Fraction
import itertools
import math
import random
import struct

import pytest
import torch

from helion._testing.cute_fragment_storage_oracle import _cpu_fma
from helion._testing.cute_fragment_storage_oracle import _fma

pytestmark = pytest.mark.cpu_only


def _fp32(bits):
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def _bits(value):
    return struct.pack("!d", value)


def test_fma_finite_fp32_operands_match_exact_product_and_double_fma():
    # Signed zero, subnormal/normal boundaries, cancellation, and products
    # spanning the FP32 exponent range. The random rows test mantissas too.
    edges = [0, 1, 0x7FFFFF, 0x800000, 0x3F800000, 0x3F800001, 0x3F7FFFFF, 0x7F7FFFFF]
    values = [_fp32(bits | sign) for bits in edges for sign in (0, 0x80000000)]
    cases = list(itertools.product(values, values, values[::4]))
    generator = random.Random(462)
    for _ in range(512):
        row = []
        while len(row) < 3:
            value = _fp32(generator.getrandbits(32))
            if math.isfinite(value):
                row.append(value)
        cases.append(tuple(row))
    native = getattr(math, "fma", None)
    for a, b, c in cases:
        exact = Fraction(a) * Fraction(b) + Fraction(c)
        expected = float(exact)
        if not exact and a * b == 0 and c == 0:
            # Round-to-nearest addition preserves -0 only for two -0 terms.
            if (
                math.copysign(1, a) * math.copysign(1, b) < 0
                and math.copysign(1, c) < 0
            ):
                expected = -0.0
        actual = _fma(a, b, c)
        assert _bits(actual) == _bits(expected), (a, b, c)
        if native is not None:
            assert _bits(actual) == _bits(native(a, b, c)), (a, b, c)


def test_cpu_fma_broadcasting_special_values_and_existing_fp32_rounding():
    a = torch.tensor([[1 + 2**-23], [-0.0], [torch.inf]], dtype=torch.float32)
    b = torch.tensor([[1 - 2**-23, 0.0, -0.0, 1.0]], dtype=torch.float32)
    c = torch.tensor([[-1.0, 0.0, -0.0, -torch.inf]], dtype=torch.float32)
    actual = _cpu_fma(
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], torch.float32, True, 1
    )
    expected = torch.tensor(
        [
            [-(2**-46), 0.0, -0.0, -torch.inf],
            [-1.0, 0.0, 0.0, -torch.inf],
            [torch.inf, torch.nan, torch.nan, torch.nan],
        ],
        dtype=torch.float32,
    )
    assert actual.shape == (3, 4) and actual.dtype == torch.float32
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite_or_inf = ~torch.isnan(expected)
    assert torch.equal(
        actual.view(torch.int32)[finite_or_inf],
        expected.view(torch.int32)[finite_or_inf],
    )
