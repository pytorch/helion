"""Tests for the CuTe provably-FTZ-safe exp marking.

``sum(exp(x - amax(x, dim)))`` — the cross-entropy / logsumexp / softmax
denominator pattern — always sums to >= 1 (the max element contributes
exp(0) = 1), so exp outputs below 2^-126, the only range where
``ex2.approx.ftz`` differs from the guarded default lowering, can never
reach the fp32 sum's ulp.  ``mark_ftz_safe_exp_nodes`` proves the pattern
on the device IR and the cute op overrides emit ``fastmath=True`` for the
marked exp sites only, independent of the ``fast_math`` setting (which
still controls every exp the proof does not cover).

Lives in ``helion/_compiler/cute/exp2_fastmath.py``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute")
def _logsumexp_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_x[:, None]), dim=-1)
        out[tile_m] = max_x + torch.log(sum_exp)
    return out


@helion.kernel(backend="cute")
def _softmax_numerator_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        max_x = torch.amax(rows, dim=-1)
        # exp escapes into the output (softmax numerator): flushing its
        # denormal values WOULD change the result, so no fastmath mark.
        exp_x = torch.exp(rows - max_x[:, None])
        out[tile_m, :] = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    return out


@helion.kernel(backend="cute")
def _shifted_by_other_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        rows = x[tile_m, :].to(torch.float32)
        other = y[tile_m, :].to(torch.float32)
        # Shift by the max of a DIFFERENT tensor: the sum >= 1 lower bound
        # does not hold, so no fastmath mark.
        max_y = torch.amax(other, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_y[:, None]), dim=-1)
        out[tile_m] = max_y + torch.log(sum_exp)
    return out


@onlyBackends(["cute"])
class TestCuteFtzSafeExp(TestCase):
    def test_logsumexp_marked(self) -> None:
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(_logsumexp_kernel, (x,))
        self.assertIn("fastmath=True", code)
        ref = torch.logsumexp(x.float(), dim=-1)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_softmax_numerator_not_marked(self) -> None:
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(_softmax_numerator_kernel, (x,))
        self.assertIn("exp2", code)
        self.assertNotIn("fastmath=True", code)
        ref = torch.softmax(x.float(), dim=-1)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_shifted_by_other_tensor_not_marked(self) -> None:
        x = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 1024, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(_shifted_by_other_kernel, (x, y))
        self.assertIn("exp2", code)
        self.assertNotIn("fastmath=True", code)
        max_y = y.float().amax(dim=-1)
        ref = max_y + torch.log(
            torch.sum(torch.exp(x.float() - max_y[:, None]), dim=-1)
        )
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    import unittest

    unittest.main()
