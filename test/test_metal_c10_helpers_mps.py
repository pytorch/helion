"""Helion's copies of c10 helpers (``helion._compiler.metal.c10_helpers``)
give torch's own bits: each copy runs beside the c10 header's on edge cases
(negatives, zeros of both signs, infinities, NaN, large and tiny
magnitudes), in each dtype Helion's kernels call it with, through raw MSL."""

from __future__ import annotations

import itertools
import sys
import unittest

import pytest
import torch

from helion._compiler.metal.c10_helpers import HELPERS
from helion._compiler.metal.c10_helpers import helpers_source
from helion._testing import skipUnlessBackends

DEVICE = "mps"

pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]

_MSL_TYPES = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint32: "uint",
}
#: Bit views for comparing outputs, NaNs included.
_BITS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}

_FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.0, 7.0, -7.0, 1e-5, 1e-30]
_FLOATS += [65504.0, 1e30, -1e30, float("inf"), float("-inf"), float("nan")]
_INTS = [0, 1, -1, 3, -3, 7, -7, 100, -100, 2**31 - 1, -(2**31) + 1]
_LONGS = [*_INTS, 2**62, -(2**62), 2**63 - 1]
_UINTS = [0, 1, 3, 7, 100, 2**31, 2**32 - 1]

#: (helper, operand dtype, result dtype): the binary helpers in the dtypes
#: Helion's kernels call them with (index math in ``int``/``long``/``uint``,
#: tensor math in the float types).
_BINARY = [
    # Not bfloat: c10's float floor_divide returns float as bfloat, which MSL
    # won't convert implicitly, so Helion computes bfloat's in float.
    *(("floor_divide", t, t) for t in (torch.float32, torch.float16)),
    *(("floor_divide", t, t) for t in (torch.int32, torch.int64, torch.uint32)),
    *(("remainder", t, torch.float32) for t in (torch.float32, torch.float16)),
    ("remainder", torch.bfloat16, torch.float32),
    *(("remainder", t, t) for t in (torch.int32, torch.int64)),
    *(("max", t, t) for t in _BITS),
    *(("max", t, t) for t in (torch.int32, torch.int64, torch.uint32)),
    *(("min", t, t) for t in _BITS),
    *(("min", t, t) for t in (torch.int32, torch.int64, torch.uint32)),
]
#: The SIMD-group reductions in the dtypes the reduction preamble
#: instantiates them with (the accumulator types: ``long`` takes the
#: headers, see ``c10_helpers``).
_SIMD = [
    (op, t)
    for op in ("simd_sum", "simd_prod", "simd_max", "simd_min")
    for t in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.uint32)
]


def _pairs(dtype: torch.dtype, nonzero_b: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Every pair of *dtype*'s edge values; for integer division, divisors
    without 0 (and no INT_MIN / -1)."""
    values = (
        _FLOATS
        if dtype.is_floating_point
        else _LONGS
        if dtype == torch.int64
        else _UINTS
        if dtype == torch.uint32
        else _INTS
    )
    divisors = [v for v in values if v != 0] if nonzero_b else values
    a, b = zip(*itertools.product(values, divisors), strict=True)
    return torch.tensor(a, dtype=dtype), torch.tensor(b, dtype=dtype)


def _rows(dtype: torch.dtype) -> torch.Tensor:
    """Rows of 32 values, one per SIMD group: random, with a NaN, with both
    infinities, of large magnitudes; integer rows random, with negatives and
    extremes."""
    gen = torch.Generator().manual_seed(0)
    if dtype.is_floating_point:
        rows = torch.randn(6, 32, generator=gen)
        rows[1, 5] = float("nan")
        rows[2, 3], rows[2, 9] = float("inf"), float("-inf")
        rows[3] *= 1e30 if dtype == torch.float32 else 1e4
        rows[4] = rows[4].abs() * 2  # products that grow
        rows[5, :8] = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 0.5, -0.5, 3.0, -3.0])
        return rows.to(dtype)
    high = 2**32 if dtype == torch.uint32 else 2**31
    low = 0 if dtype == torch.uint32 else -(2**31)
    rows = torch.randint(low, high, (6, 32), generator=gen, dtype=torch.int64)
    rows[4] = torch.randint(-3 if low else 0, 4, (32,), generator=gen)
    return rows.to(dtype)


def _source(c10: list[str]) -> str:
    """One library holding a kernel per binary case and per SIMD case,
    behind *c10*'s lines."""
    kernels = []
    for i, (name, operand, result) in enumerate(_BINARY):
        t, r = _MSL_TYPES[operand], _MSL_TYPES[result]
        kernels.append(
            f"kernel void binary_{i}(device const {t}* a [[buffer(0)]], "
            f"device const {t}* b [[buffer(1)]], device {r}* out [[buffer(2)]], "
            "uint i [[thread_position_in_grid]]) {\n"
            f"  out[i] = static_cast<{r}>(c10::metal::{name}(a[i], b[i]));\n}}"
        )
    for i, (name, dtype) in enumerate(_SIMD):
        t = _MSL_TYPES[dtype]
        kernels.append(
            f"kernel void simd_{i}(device const {t}* a [[buffer(0)]], "
            f"device {t}* out [[buffer(1)]], "
            "uint i [[thread_position_in_grid]]) {\n"
            f"  out[i] = c10::metal::{name}(a[i]);\n}}"
        )
    kernels.append(
        "kernel void width(device uint* out [[buffer(0)]], "
        "uint i [[thread_position_in_grid]]) {\n"
        "  out[i] = c10::metal::simdgroup_size;\n}"
    )
    return "\n".join(
        ["#include <metal_stdlib>", *c10, "using namespace metal;", *kernels]
    )


def _same_bits(
    test: unittest.TestCase, ours: torch.Tensor, theirs: torch.Tensor
) -> None:
    bits = _BITS.get(ours.dtype)
    if bits is not None:
        ours, theirs = ours.view(bits), theirs.view(bits)
    test.assertTrue(torch.equal(ours.cpu(), theirs.cpu()))


@unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)
class TestC10HelpersMps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ours = torch.mps.compile_shader(_source(helpers_source(set(HELPERS))))
        cls.theirs = torch.mps.compile_shader(
            _source(
                [
                    "#include <c10/metal/utils.h>",
                    "#include <c10/metal/reduction_utils.h>",
                ]
            )
        )

    def test_binary(self) -> None:
        """floor_divide, remainder, max and min over every pair of edge
        values."""
        for i, (name, operand, result) in enumerate(_BINARY):
            with self.subTest(name=name, dtype=operand):
                division = name in ("floor_divide", "remainder")
                a, b = (
                    t.to(DEVICE)
                    for t in _pairs(operand, division and not operand.is_floating_point)
                )
                outs = []
                for library in (self.ours, self.theirs):
                    out = torch.empty(a.numel(), dtype=result, device=DEVICE)
                    getattr(library, f"binary_{i}")(a, b, out, threads=a.numel())
                    outs.append(out)
                _same_bits(self, *outs)

    def test_simd(self) -> None:
        """The SIMD-group reductions over rows of edge values, one SIMD
        group per row."""
        for i, (name, dtype) in enumerate(_SIMD):
            with self.subTest(name=name, dtype=dtype):
                a = _rows(dtype).to(DEVICE).flatten()
                outs = []
                for library in (self.ours, self.theirs):
                    out = torch.empty_like(a)
                    getattr(library, f"simd_{i}")(
                        a, out, threads=a.numel(), group_size=32
                    )
                    outs.append(out)
                _same_bits(self, *outs)

    def test_simdgroup_size(self) -> None:
        outs = []
        for library in (self.ours, self.theirs):
            out = torch.empty(1, dtype=torch.uint32, device=DEVICE)
            library.width(out, threads=1)
            outs.append(out)
        _same_bits(self, *outs)


if __name__ == "__main__":
    unittest.main()
