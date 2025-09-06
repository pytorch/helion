"""
Root Mean Square Normalization Example
=================================

This example demonstrates how to implement a Root Mean Square (RMS) normalization
operation using Helion.
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
# RMS Normalization Kernel
# ---------------------
@helion.kernel(static_shapes=True)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Performs Root Mean Square (RMS) normalization on the input tensor.

    RMS normalization normalizes by the root mean square of the elements:
    output = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor of shape [M, N]
        weight: Scale parameter of shape [N]
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape [M, N] with RMS normalization applied
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)

        # Compute RMS: sqrt(mean(x^2))
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_x_squared + eps)

        # Apply normalization and weight
        normalized = x_tile * rms
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)

    return out


# %%
# Benchmark Wrapper
# --------------
def rms_norm_tritonbench(H: int, inp: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for tritonbench that matches expected interface.

    Args:
        H: Hidden dimension size
        inp: Input tensor

    Returns:
        Normalized tensor
    """
    weight = torch.ones(H, device=inp.device, dtype=inp.dtype)
    return rms_norm(inp, weight, eps=1e-6)


# %%
# Reference Implementation
# --------------------
def rms_norm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the RMS norm kernel implementation against the PyTorch reference implementation.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    weight = torch.randn([n], device="cuda", dtype=torch.float16)
    run_example(rms_norm, rms_norm_pytorch, (x, weight, 1e-5))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the RMS norm kernel verification with different tensor sizes.

    Tests with three configurations:
    - 32x64
    - 128x256
    - 1024x1024
    - 2048x1024
    """
    check(32, 64)
    check(128, 256)
    check(1024, 1024)
    check(2048, 1024)


if __name__ == "__main__":
    main()
