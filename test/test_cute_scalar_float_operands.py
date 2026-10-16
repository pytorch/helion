"""A Python-float host scalar (a SymFloat such as ``1.0 / n``) in CuTe math.

The DSL types ``n + 0.0`` (an Int shape argument converted to float) as
Float64, so the scalar used to turn float32 tensor math into Float64: a loop
carry changed type (``TYPE_UNSTABLE_JOIN``) and comparisons ran in double.
Torch rounds such a scalar to the op's float32 math, and keeps it in double
for float64 math.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(static_shapes=False, autotune_effort="none")
def scalar_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([], dtype=x.dtype)
        for _ in hl.tile(n, block_size=16):
            total = total + 1.0 / n
        out[tile_m] = x[tile_m, 0] + total
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def row_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=16):
            acc = acc + x[tile_m, tile_n].sum(-1) * (1.0 / n) + 1.0 / (n + 3)
        out[tile_m] = acc
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def above_scalar(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.where(
            x[tile_m, tile_n] > 1.0 / (n - 61), 1.0, 0.0
        ).to(x.dtype)
    return out


def _row_carry_reference(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    acc = torch.zeros(m, dtype=x.dtype, device=x.device)
    for start in range(0, n, 16):
        acc = acc + x[:, start : start + 16].sum(-1) * (1.0 / n) + 1.0 / (n + 3)
    return acc


@onlyBackends(["cute"])
class TestCuteScalarFloatOperands(TestCase):
    def test_float_carries_keep_their_dtype(self) -> None:
        """``total + 1.0 / n`` keeps a float32 (or float64) carry's type, and
        each step rounds the scalar like torch does."""
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(64, 64, device=DEVICE, dtype=dtype)
            with self.subTest(dtype=dtype):
                _, out = code_and_output(scalar_carry, (x,))
                total = torch.zeros((), dtype=dtype, device=DEVICE)
                for _ in range(64 // 16):
                    total = total + 1.0 / 64
                torch.testing.assert_close(out, x[:, 0] + total, rtol=0, atol=0)
                _, out = code_and_output(row_carry, (x,))
                torch.testing.assert_close(out, _row_carry_reference(x), rtol=0, atol=0)

    def test_comparison_rounds_the_scalar_to_float32(self) -> None:
        """``x > 1.0 / 3`` on float32 ``x`` compares against the scalar
        rounded to float32, so float32(1/3) is not above it; in float64 the
        scalar keeps its double value."""
        for dtype in (torch.float32, torch.float64):
            x = torch.full((64, 64), 1.0 / 3, device=DEVICE, dtype=dtype)
            x[:, 1::2] = torch.nextafter(x[:, 1::2], torch.ones_like(x[:, 1::2]))
            with self.subTest(dtype=dtype):
                _, out = code_and_output(above_scalar, (x,))
                expected = torch.where(x > 1.0 / 3, 1.0, 0.0).to(dtype)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
