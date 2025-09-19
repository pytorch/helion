"""
Low mem dropout Example
================

This example demonstrates how to implement a Low mem dropout using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Low mem dropout implementations
# -------------------
@helion.kernel()
def low_mem_dropout(p: float, x: torch.Tensor, x_keep: torch.Tensor) -> torch.Tensor:
    """
    Applies dropout on x using p
    Args:
        p (float): dropout probability
        x (torch.Tensor): input tensor
        x_keep (torch.Tensor): mask tensor indicating which elements to keep
    Returns:
        Output tensor
    """

    scale = 1.0 / (1.0 - p)
    # flatten to 1D so we can use tile
    n = x.numel()
    x_flat, m_flat = x.view(-1), x_keep.view(-1)
    out_flat = torch.empty_like(x_flat)

    for tidx in hl.tile(n):
        xi = x_flat[tidx].to(torch.float32)
        mi = m_flat[tidx].to(torch.float32) > 0.5
        yscaled = xi * scale
        zeros = xi - xi
        yi = torch.where(mi, yscaled, zeros)
        out_flat[tidx] = yi.to(x.dtype)
    return out_flat.view_as(x)


# %%
# TritonBench Wrapper
# -------------------
def low_mem_dropout_tritonbench(tb_op: object, p: float, x: torch.Tensor) -> Callable:
    """
    Wrapper for TritonBench compatibility.

    Args:
        tb_op: TritonBench operator instance
        p (float): dropout probability
        x (torch.Tensor): Input tensor

    Returns:
        Callable: A function that performs the low_mem_dropout.
    """
    torch.manual_seed(123)  # Set seed for reproducibility
    x_keep = torch.rand_like(x) > p
    return lambda: low_mem_dropout(p, x, x_keep)


# %%
# Baseline Function
# -------------------
def eager_dropout(p: float, x: torch.Tensor, x_keep: torch.Tensor) -> torch.Tensor:
    return x * x_keep.to(x.dtype) / (1 - p)


# %%
# Verification Function
# -------------------
def check(p: float, size: int) -> None:
    """
    Verify the low mem dropout kernel implementation against PyTorch's native dropout implementation.

    Args:
        p (float): dropout probability
        size (int): input tensor size
    """
    x = torch.randn(size=(size,)).cuda()
    torch.manual_seed(123)  # Set seed for reproducibility
    x_keep = torch.rand_like(x) > p
    kernels = {"low_mem_dropout": low_mem_dropout}
    run_example(kernels, eager_dropout, (p, x, x_keep))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the low mem dropout kernel verification with different tensor sizes.

    Tests with two configurations:
    - p=0.25, s=8192
    - p=0.25, s=32768
    """
    check(0.25, 8192)
    check(0.25, 32768)


if __name__ == "__main__":
    main()
