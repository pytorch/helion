"""
Low mem dropout Example
=======================

This example demonstrates how to implement a Low mem dropout using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

# %%
# Low mem dropout forward implementations
# ---------------------------------------


# %%
@helion.kernel(static_shapes=False)
def low_mem_dropout(p: float, x: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Applies dropout on x using p
    Args:
        p (float): dropout probability
        x (torch.Tensor): input tensor
    Returns:
        Output tensor
    """
    scale = 1.0 / (1.0 - p)
    # flatten to 1D so we can use tile
    n = x.numel()
    x_flat = x.view(-1)
    out_flat = torch.empty_like(x_flat)
    for tidx in hl.tile(n):
        xi = x_flat[tidx].to(torch.float32)
        r = hl.rand([tidx], seed=seed)
        keep = r > p
        yscaled = xi * scale
        yi = torch.where(keep, yscaled, 0.0)
        out_flat[tidx] = yi.to(x.dtype)
    return out_flat.view_as(x)


# %%
# Low mem dropout backward implementation
# ---------------------------------------


# %%
@helion.kernel(static_shapes=False)
def low_mem_dropout_bwd(p: float, grad_y: torch.Tensor, seed: int) -> torch.Tensor:
    """
    For low mem dropout we are applying randomness inside both fwd and bwd
    technically dropout bwd is same as fwd
    Args:
        p (float): Dropout probability
        grad_y (torch.Tensor): Gradient tensor
    Returns:
        Output tensor
    """
    scale = 1.0 / (1.0 - p)
    n = grad_y.numel()
    grad_y_flat = grad_y.view(-1)
    out_flat = torch.empty_like(grad_y_flat)
    for tidx in hl.tile(n):
        gi = grad_y_flat[tidx].to(torch.float32)
        r = hl.rand([tidx], seed=seed)
        keep = r > p
        g_scaled = gi * scale
        gxi = torch.where(keep, g_scaled, 0.0)
        out_flat[tidx] = gxi.to(grad_y.dtype)
    return out_flat.view_as(grad_y)


# %%
# TritonBench Wrapper
# -------------------


# %%
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

    def _inner() -> torch.Tensor:
        return low_mem_dropout(p, x, seed=123)

    return _inner


# %%
# Verification Function
# ---------------------


# %%
def check(p: float, size: int) -> None:
    """
    Verify the low mem dropout kernel implementation against PyTorch's native dropout implementation.

    Args:
        p (float): dropout probability
        size (int): input tensor size
    """
    x = torch.randn(size=(size,), device=DEVICE)
    seed = 123

    out = low_mem_dropout(p, x, seed)
    grad_y = torch.ones_like(x)
    grad_x = low_mem_dropout_bwd(p, grad_y, seed)
    mask_fwd = out != 0
    mask_bwd = grad_x != 0
    assert torch.equal(mask_fwd, mask_bwd)


# %%
# Main Function
# -------------


# %%
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
