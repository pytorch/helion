"""
Matrix Multiplication with Layer Normalization Example
======================================================

This example demonstrates how to implement a fused matrix multiplication and layer normalization
operation using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# MatMul-LayerNorm Kernel
# -----------------------
# static_shapes=True gives a performance boost for matmuls


# %%
@helion.kernel(static_shapes=True)
def matmul_layernorm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Performs matrix multiplication followed by layer normalization.

    Args:
        x: First input tensor of shape [M, K]
        y: Second input tensor of shape [K, N]
        weight: Layer normalization weight parameter of shape [N]
        bias: Layer normalization bias parameter of shape [N]

    Returns:
        Output tensor of shape [M, N] containing the result of matrix multiplication followed by layer normalization
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            mm = torch.matmul(x[tile_m, tile_k], y[tile_k, :])
            acc = acc + mm
        eps = 1e-5
        sum_vals = acc.sum(dim=-1, keepdim=True)
        mean = sum_vals / n
        centered = acc - mean
        var = (centered * centered).sum(dim=-1, keepdim=True) / n
        normalized = centered * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))
        out[tile_m, :] = acc
    return out


# %%
# Reference Implementation
# ------------------------


# %%
def matmul_layernorm_pytorch(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch reference implementation of matrix multiplication followed by layer normalization.

    Args:
        x: First input tensor of shape [M, K]
        y: Second input tensor of shape [K, N]
        weight: Layer normalization weight parameter of shape [N]
        bias: Layer normalization bias parameter of shape [N]

    Returns:
        Output tensor of shape [M, N] containing the result of matrix multiplication followed by layer normalization
    """
    matmul_out = torch.matmul(x, y)

    ln_out = F.layer_norm(
        matmul_out.to(torch.float32),
        normalized_shape=(matmul_out.shape[-1],),
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
    )

    return ln_out.to(torch.promote_types(x.dtype, y.dtype))


# %%
# Verification Function
# ---------------------


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Verify the matmul_layernorm kernel implementation against the PyTorch reference implementation.

    Args:
        m: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        n: Second dimension of the second matrix
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    weight = torch.randn([n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    run_example(matmul_layernorm, matmul_layernorm_pytorch, (x, y, weight, bias))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the matmul_layernorm kernel verification with different matrix sizes.

    Tests with two configurations:
    - 32x64 * 64x200
    - 128x256 * 256x400
    """
    # TODO(yf225): n=64 or 128 throws error, need to investigate
    # check(32, 64, 64)
    # check(32, 64, 128)
    check(32, 64, 200)
    check(128, 256, 400)


if __name__ == "__main__":
    main()
