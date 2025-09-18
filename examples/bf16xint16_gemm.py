"""
BFloat16 × Int16 GEMM Example
==============================
This example implements a Helion kernel that multiplies a bf16 activation matrix by an int16
weight matrix. The kernel mirrors TritonBench's optimized bf16xint16 GEMM by explicitly casting
whichever operand is int16 to bf16 before performing a float32-accumulated matmul and returning
bf16 results. It includes utilities for correctness checking and TritonBench integration.
"""

from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(static_shapes=True)
def bf16xint16_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute x @ w where one operand may be int16 and the other bf16.

    Both operands are treated as 2-D matrices. Any int16 operand is cast to bf16 on the fly.
    Accumulation happens in float32 for numerical accuracy, and the final result is bf16.
    """
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError("bf16xint16_gemm expects 2-D inputs")
    m, k = x.size()
    k2, n = w.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            x_tile = x[tile_m, tile_k].to(torch.bfloat16)
            w_tile = w[tile_k, tile_n].to(torch.bfloat16)
            acc = torch.addmm(acc, x_tile, w_tile)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def reference_bf16xint16_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Reference implementation that casts any int16 operand to bf16 before matmul."""
    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    w_bf16 = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
    return torch.matmul(x_bf16, w_bf16)


def check(m: int, k: int, n: int, transpose: bool = False) -> None:
    """Validate the kernel against the reference implementation."""
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randint(
        -(2**15),
        2**15 - 1,
        (k, n),
        device="cuda",
        dtype=torch.int16,
    )

    if transpose:
        lhs = w.T.contiguous()
        rhs = x.T.contiguous()
    else:
        lhs, rhs = x, w

    run_example(bf16xint16_gemm, reference_bf16xint16_gemm, (lhs, rhs))


def bf16xint16_gemm_tritonbench(
    tb_op: object, x: torch.Tensor, w: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """Adapter used by benchmarks/run.py to match TritonBench's operator API."""
    x_mat = x.reshape(-1, x.size(-1))
    return lambda: bf16xint16_gemm(x_mat, w)


def main() -> None:
    """Run a quick correctness sweep."""
    check(1024, 2048, 1024)
    check(1024, 2048, 1024, transpose=True)


if __name__ == "__main__":
    main()
