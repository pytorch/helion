"""
Softmax Function Example
===================

This example demonstrates how to implement softmax operations using different approaches in Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Simple Softmax Kernel
# -----------------
@helion.kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Performs softmax operation along dimension 1 using PyTorch's built-in softmax.

    Args:
        x: Input tensor of shape [N, M]

    Returns:
        Output tensor of shape [N, M] with softmax applied along dimension 1
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# %%
# Decomposed Softmax Kernel
# ---------------------
@helion.kernel()
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    """
    Performs softmax operation along dimension 1 using manual decomposition.

    Implements the softmax algorithm step by step:
    1. Find the maximum value for numerical stability
    2. Subtract the maximum and compute exponentials
    3. Normalize by the sum of exponentials

    Args:
        x: Input tensor of shape [N, M]

    Returns:
        Output tensor of shape [N, M] with softmax applied along dimension 1
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


# %%
# Two-Pass Optimized Softmax Kernel
# -----------------------------
@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    """
    Performs softmax operation in two passes for better performance.

    This optimized version computes softmax with fewer passes over the data,
    trading some numerical stability for performance.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M, N] with softmax applied along dimension 1
    """
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the softmax kernel implementations against PyTorch's native softmax function.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    kernels = {
        "helion simple": softmax,
        # "helion decomposed": softmax_decomposed,
        "helion two pass": softmax_two_pass,
    }
    run_example(kernels, lambda x: torch.nn.functional.softmax(x, dim=1), (x,))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the softmax kernel verification with a 1024x1024 tensor.
    """
    check(1024, 1024)


if __name__ == "__main__":
    main()
