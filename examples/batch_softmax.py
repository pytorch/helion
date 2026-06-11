"""
Batch Softmax Example (Arithmetic Broadcasting)
================================================

This example demonstrates arithmetic broadcasting in Helion kernels via a
batched softmax: x[B, M, N] -> softmax over the last dimension.

The batch dims [B, M] are tiled directly with ``hl.tile([b, m])`` and the last
dim is loaded whole, so the kernel works in the native 3D layout with no reshape
— which also avoids a retile of the output on the TPU/Pallas backend. The
[:, :, None] pattern broadcasts the reduced [tile_b, tile_m] back over the last
dim.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Batch Softmax Kernel
# --------------------


# %%
@helion.kernel(
    # Validate autotuning against eager softmax (the ground truth) rather than
    # the kernel's default config, which loads the whole last dim and so can be
    # too large to compile as the baseline before autotuning shrinks the block.
    autotune_baseline_fn=lambda x: torch.nn.functional.softmax(x, dim=-1),
)
def batch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Batched softmax with arithmetic broadcasting.

    Args:
        x: Input tensor of shape [B, M, N]

    Returns:
        Softmax output of shape [B, M, N], normalized along the last dimension.
    """
    b, m, n = x.size()
    out = torch.empty_like(x)
    # Tile the batch dims [B, M] together; load the whole last dim per tile.
    for tile_b, tile_m in hl.tile([b, m]):
        values = x[tile_b, tile_m, :]  # [tile_b, tile_m, N]

        # Reduce over last dim -> [tile_b, tile_m]
        x_max = torch.amax(values, dim=2)

        # Broadcast x_max from [tile_b, tile_m] to [tile_b, tile_m, 1]
        # using [:, :, None], then subtract from [tile_b, tile_m, N]
        exp_vals = torch.exp(values - x_max[:, :, None])

        sum_exp = torch.sum(exp_vals, dim=2)  # [tile_b, tile_m]
        out[tile_b, tile_m, :] = exp_vals / sum_exp[:, :, None]
    return out


# %%
# Verification Function
# ---------------------


# %%
def check(b: int, m: int, n: int) -> None:
    x = torch.randn([b, m, n], device=DEVICE, dtype=HALF_DTYPE)
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
