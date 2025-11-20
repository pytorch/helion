from __future__ import annotations

import torch
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


@triton.jit
def add_pairs(a, b):
    return a + b


@triton.jit
def pairwise_ops(a, b):
    sum_val = a + b
    prod_val = a * b
    return sum_val, prod_val


@triton.jit
def vector_mix_norm(a, b):
    mixed = tl.where(a > b, a - b, a + b)
    l2sq = tl.sum(mixed * mixed)
    inv = tl.rsqrt(tl.maximum(l2sq, 1e-12))
    return mixed * inv


@helion.kernel(autotune_effort="none")
def triton_kernel_add_pairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        x_val = x[tile]
        y_val = y[tile]
        # Pass by function name string to avoid closures in kernels
        result = hl.triton_kernel("add_pairs", args=(x_val, y_val), output_like=x_val)
        out[tile] = result
    return out


class TestTritonKernel(RefEagerTestDisabled, TestCase):
    def test_triton_kernel_simple_add(self) -> None:
        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(triton_kernel_add_pairs, (x, y))
        self.assertIn("@triton.jit", code)
        self.assertIn("add_pairs", code)
        torch.testing.assert_close(result, x + y)
        self.assertExpectedJournal(code)

    def test_triton_kernel_multi_output(self) -> None:
        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sum_out = torch.empty_like(x)
            prod_out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                sum_val, prod_val = hl.triton_kernel(
                    "pairwise_ops",
                    args=(x_val, y_val),
                    output_like=(x_val, x_val),
                )
                sum_out[tile] = sum_val
                prod_out[tile] = prod_val
            return sum_out, prod_out

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, (sum_result, prod_result) = code_and_output(k, (x, y))
        self.assertIn("pairwise_ops", code)
        torch.testing.assert_close(sum_result, x + y)
        torch.testing.assert_close(prod_result, x * y)
        self.assertExpectedJournal(code)

    def test_triton_kernel_tl_ops(self) -> None:
        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                out[tile] = hl.triton_kernel(
                    "vector_mix_norm",
                    args=(x_val, y_val),
                    output_like=x_val,
                )
            return out

        x = torch.randn(96, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(k, (x, y))
        self.assertIn("vector_mix_norm", code)

        bs = 32
        expected = torch.empty_like(x)
        for i in range(0, x.numel(), bs):
            xa = x[i : i + bs]
            ya = y[i : i + bs]
            mixed = torch.where(xa > ya, xa - ya, xa + ya)
            l2sq = torch.sum(mixed * mixed)
            inv = torch.rsqrt(
                torch.maximum(l2sq, torch.tensor(1e-12, device=DEVICE, dtype=x.dtype))
            )
            expected[i : i + bs] = mixed * inv
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    import unittest

    unittest.main()
