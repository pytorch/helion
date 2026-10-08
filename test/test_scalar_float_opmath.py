"""A Python host scalar (a SymFloat or SymInt) in fp16/bf16 arithmetic.

Torch computes ``x * s``, ``x + s``, ``s - x`` on a 16-bit ``x`` at float32
opmath with the scalar at float32, rounding only the result.  A SymFloat
operand (``1.0 / n`` with dynamic shapes, or a float kernel argument) or a
SymInt operand (``n``, or an int kernel argument) was rounded to the 16-bit
dtype first, so these results differed from eager, and from a static-shape
literal scalar, in many elements.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
import helion.language as hl


@helion.kernel(static_shapes=False, autotune_effort="none")
def scalar_arith(x: torch.Tensor, scale: float) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([5, m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        v = x[tile_m, tile_n]
        out[0, tile_m, tile_n] = v * (1.0 / n)
        out[1, tile_m, tile_n] = v + 7.3 / n
        out[2, tile_m, tile_n] = 7.3 / n - v
        out[3, tile_m, tile_n] = v * scale
        out[4, tile_m, tile_n] = v * scale + v * scale
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def scalar_carry(x: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=16):
            acc = acc + eps
            acc = acc + x[tile_m, tile_n].sum(-1) * (1.0 / n)
        out[tile_m] = acc
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def int_scalar_arith(x: torch.Tensor, k: int) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([5, m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        v = x[tile_m, tile_n]
        out[0, tile_m, tile_n] = v * k
        out[1, tile_m, tile_n] = v + k
        out[2, tile_m, tile_n] = (n + 2000) - v
        out[3, tile_m, tile_n] = v / n
        out[4, tile_m, tile_n] = v * k + v * k
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def reciprocal_then_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([2, m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        v = x[tile_m, tile_n]
        out[0, tile_m, tile_n] = torch.reciprocal(v) * w[tile_m, tile_n]
        out[1, tile_m, tile_n] = 7.3 / v
    return out


def _scalar_arith_reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    n = x.shape[1]
    return torch.stack(
        [
            x * (1.0 / n),
            x + 7.3 / n,
            7.3 / n - x,
            x * scale,
            x * scale + x * scale,
        ]
    )


@onlyBackends(["triton", "cute"])
class TestScalarFloatOpmath(TestCase):
    def test_scalar_stays_float32_in_half_math(self) -> None:
        """Each op rounds only its result to the 16-bit dtype, bit for bit
        like eager, both for ``1.0 / n`` and for a float kernel argument."""
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(40, 111, device=DEVICE, dtype=dtype) * 3
            with self.subTest(dtype=dtype):
                _, out = code_and_output(scalar_arith, (x, 0.0657))
                torch.testing.assert_close(
                    out, _scalar_arith_reference(x, 0.0657), rtol=0, atol=0
                )

    def test_half_carry_keeps_its_dtype(self) -> None:
        """A float32 scalar added to a 16-bit loop carry leaves the carry
        16-bit, rounded at each step like eager."""
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(64, 64, device=DEVICE, dtype=dtype)
            with self.subTest(dtype=dtype):
                _, out = code_and_output(scalar_carry, (x, 0.013))
                acc = torch.zeros(64, dtype=dtype, device=DEVICE)
                for start in range(0, 64, 16):
                    acc = acc + 0.013
                    acc = acc + x[:, start : start + 16].sum(-1) * (1.0 / 64)
                torch.testing.assert_close(out, acc, rtol=0, atol=0)

    def test_int_scalar_stays_float32_in_half_math(self) -> None:
        """An int kernel argument or a dynamic size computes at float32 like
        eager, so 2049 is not rounded to 2048 in fp16 (257 to 256 in bf16)."""
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(40, 257, device=DEVICE, dtype=dtype) * 3
            with self.subTest(dtype=dtype):
                _, out = code_and_output(int_scalar_arith, (x, 2049))
                expected = torch.stack(
                    [x * 2049, x + 2049, 2257 - x, x / 257, x * 2049 + x * 2049]
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @skipIfXPU("XPU 16-bit reciprocal results differ from XPU eager")
    @skipIfRocm("ROCm 16-bit reciprocal results differ from ROCm eager")
    def test_half_reciprocal_is_rounded(self) -> None:
        """``reciprocal`` rounds to the 16-bit dtype before the next op, as
        does ``s / x``, which traces through it."""
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(40, 111, device=DEVICE, dtype=dtype) * 2
            w = torch.randn(40, 111, device=DEVICE, dtype=dtype)
            with self.subTest(dtype=dtype):
                _, out = code_and_output(reciprocal_then_mul, (x, w))
                expected = torch.stack([torch.reciprocal(x) * w, 7.3 / x])
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
