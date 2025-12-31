"""
Helion Split-K Matmul with Barrier Example
==========================================
This example demonstrates a two-stage split-K matrix multiplication using
hl.barrier() for grid-wide synchronization. The barrier approach ensures
deterministic results as opposed to atomic_add approaches.

Stage 1: Split K dimension into chunks and compute partial products
Barrier: Grid-wide synchronization to ensure all partials are written
Stage 2: Reduce partials across the split dimension
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl


@helion.kernel(static_shapes=True, dot_precision="ieee")
def split_k_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Two-stage split-K matmul using hl.barrier().  The barrier approach
    gives deterministic results as opposed to the atomic_add approach.

    Stage 1:
      - Split K into `split_k` contiguous chunks.
      - Each chunk computes a partial [tile_m, tile_n] product into its own slice of `tmp`.

    Barrier:
      - Grid-wide barrier to ensure all partials are written before reduction.

    Stage 2:
      - Reduce partials across the split dimension and write `out`.

    Shapes:
      a: [M, K]
      b: [K, N]
      tmp: [M, N, split_k]
      out: [M, N]

    Notes:
      - Static shapes keep codegen simpler.
      - `split_k` is fixed for clarity; autotuning could choose it instead.
    """
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    for tile_m, tile_n, tile_k_outer in hl.tile(
        [m, n, k], block_size=[None, None, block_k]
    ):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        # this could be a hl.atomic_add to avoid the barrier, but that would be non-determinstic
        tmp[tile_m, tile_n, tile_k_outer.id] = acc

    hl.barrier()

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)

    return out


def check(m: int, k: int, n: int) -> None:
    a = torch.randn(m, k, device=DEVICE)
    b = torch.randn(n, k, device=DEVICE).T

    run_example(
        split_k_matmul,
        torch.matmul,
        args=(a, b),
        atol=5e-1,  # long reduction accumulate errors
    )


def main() -> None:
    torch.manual_seed(0)
    check(16, 4096, 16)


if __name__ == "__main__":
    main()
