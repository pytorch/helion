"""
Broadcast Batch Matmul Example
==============================

This example demonstrates batch matrix multiplication with broadcasting:
X[B, M, K] @ W[K, N] -> [B, M, N], where W does not have a batch dimension.

The key insight is to flatten [B, M] into one dimension, reducing to a
standard 2D matmul with torch.addmm, then reshape back. torch.baddbmm
requires matching batch dims on both operands, so we cannot use it directly
when W has no batch dimension.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Broadcast Batch Matmul Kernel
# -----------------------------


# %%
@helion.kernel(static_shapes=True)
def broadcast_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiplication with broadcasting.

    Args:
        x: Input tensor of shape [B, M, K]
        w: Weight tensor of shape [K, N] (no batch dimension)

    Returns:
        Output tensor of shape [B, M, N]
    """
    b, m, k = x.size()
    k2, n = w.size()
    assert k == k2
    # Flatten [B, M, K] -> [B*M, K] to use standard 2D matmul
    x_2d = x.reshape([b * m, k])
    out_2d = torch.empty(
        [b * m, n], device=x.device, dtype=torch.promote_types(x.dtype, w.dtype)
    )
    for tile_bm, tile_n in hl.tile([b * m, n]):
        acc = hl.zeros([tile_bm, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x_2d[tile_bm, tile_k], w[tile_k, tile_n])
        out_2d[tile_bm, tile_n] = acc
    return out_2d.view(b, m, n)


# %%
# Verification Function
# ---------------------


# %%
def check(b: int, m: int, k: int, n: int) -> None:
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    w = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    # torch.matmul handles the broadcasting automatically
    run_example(broadcast_matmul, torch.matmul, (x, w))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    check(16, 512, 768, 1024)


if __name__ == "__main__":
    main()
