"""
Welford Example
===============

This example demonstrates how to implement a welford layernorm using Helion.
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
# Welford Kernel Implementations
# ------------------------------


# %%
@helion.kernel()
def welford(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    """
    Applies LayerNorm using Welford's algorithm for mean/variance.
    Args:
        weight: weight tensor of shape [N]
        bias: bias tensor of shape [N]
        x: input tensor of shape [M, N]
    Returns:
        Output tensor of shape [M, N]
    """
    m, n = x.size()

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        acc_cnt = torch.zeros_like(x[tile_m, 0], dtype=torch.float32)
        acc_mean = torch.zeros_like(acc_cnt)
        acc_m2 = torch.zeros_like(acc_cnt)

        for tile_n in hl.tile(n):
            chunk = x[tile_m, tile_n]
            Tn = chunk.size(-1)
            sum_x = torch.sum(chunk, dim=-1)
            sum_x2 = torch.sum(chunk * chunk, dim=-1)
            mean_c = sum_x / Tn
            m2_c = sum_x2 - (sum_x * sum_x) / Tn

            delta = mean_c - acc_mean
            new_cnt = acc_cnt + Tn
            new_mean = acc_mean + delta * (Tn / new_cnt)
            new_m2 = acc_m2 + m2_c + delta * delta * (acc_cnt * Tn / new_cnt)

            acc_cnt, acc_mean, acc_m2 = new_cnt, new_mean, new_m2

        rstd_tile = torch.rsqrt(acc_m2 / acc_cnt + eps)
        mean_col = acc_mean[:, None]
        rstd_col = rstd_tile[:, None]

        for tile_n in hl.tile(n):
            xi_chuck = x[tile_m, tile_n]
            w_chuck = weight[tile_n][None, :]
            b_chuck = bias[tile_n][None, :]

            y = (xi_chuck - mean_col) * rstd_col
            y = y * w_chuck + b_chuck

            out[tile_m, tile_n] = y.to(x.dtype)
    return out


# %%
# Baseline Function
# -----------------


# %%
def eager_layer_norm(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


# %%
# Verification Function
# ---------------------


# %%
def check(s: int, d: int) -> None:
    """
    Verify the welford kernel implementation against PyTorch's native layer_norm function.

    Args:
        s: First dimension of the test tensor
        d: Second dimension of the test tensor
    """

    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

    kernels = {"helion": welford}
    run_example(kernels, eager_layer_norm, (weight, bias, x))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the welford kernel verification with different tensor sizes.
    """
    check(262144, 1024)
    check(262144, 1536)
    check(262144, 2048)


if __name__ == "__main__":
    main()
