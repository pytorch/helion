"""
Batch Softmax Example (Arithmetic Broadcasting)
================================================

This example demonstrates arithmetic broadcasting in Helion kernels via a
batched softmax: x[B, M, N] -> softmax over the last dimension.

The key pattern is x - x_max, where x is [B*M, N] and x_max is [B*M].
Broadcasting uses [:, None] to expand x_max from [tile_bm] to [tile_bm, 1]
for element-wise subtraction with [tile_bm, N].

For 3D broadcasting without flattening (e.g., inside an attention kernel
where batch dims are already tiled separately), see examples/attention.py
which uses the [:, :, None] pattern: m_ij[:, :, None] broadcasts
[tile_b, tile_m] to [tile_b, tile_m, 1].
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
# Batch Softmax Kernel
# --------------------


# %%
@helion.kernel()
def batch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Batched softmax with arithmetic broadcasting.

    Args:
        x: Input tensor of shape [B, M, N]

    Returns:
        Softmax output of shape [B, M, N], normalized along the last dimension.
    """
    b, m, n = x.size()
    # Flatten [B, M, N] -> [B*M, N] to use standard 2D softmax pattern
    x_2d = x.reshape([b * m, n])
    out_2d = torch.empty_like(x_2d)
    for tile_bm in hl.tile(b * m):
        # Load entire last dimension for each row
        values = x_2d[tile_bm, :]  # [tile_bm, N]

        # Reduce over last dim -> [tile_bm]
        x_max = torch.amax(values, dim=1)

        # Broadcast x_max from [tile_bm] to [tile_bm, 1]
        # using [:, None], then subtract from [tile_bm, N]
        exp_vals = torch.exp(values - x_max[:, None])

        sum_exp = torch.sum(exp_vals, dim=1)  # [tile_bm]
        out_2d[tile_bm, :] = exp_vals / sum_exp[:, None]
    return out_2d.view(b, m, n)


# %%
# Verification Function
# ---------------------


# %%
def check(b: int, m: int, n: int) -> None:
    x = torch.randn([b, m, n], device=DEVICE, dtype=torch.float16)
    run_example(
        batch_softmax,
        lambda x: torch.nn.functional.softmax(x, dim=-1),
        (x,),
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    check(16, 512, 1024)


if __name__ == "__main__":
    main()
