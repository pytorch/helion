"""
Helion SE Block Example
============================
This example demonstrates a Helion kernel implementation of SE Block.
"""

# %%
from __future__ import annotations

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def se_block_fwd(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """
    Performs 2 * x * sigmoid(x @ w)
    Args:
        x: 2D tensor of shape [m, n].
        w: 2D tensor of shape [n, n].
    Returns:
        out: Resulting matrix of shape [m, n].
        s: sigmoid(x @ w) of shape [m, n].
    """
    m, n = x.size()

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            # Compute sigmoid in float32
            sigmoid_result = torch.sigmoid(x[tile_m, :] @ w[:, tile_n])
            s[tile_m, tile_n] = sigmoid_result
            # Compute output: 2 * x * sigmoid, cast to input dtype
            acc = 2.0 * x[tile_m, tile_n].to(torch.float32) * sigmoid_result
            out[tile_m, tile_n] = acc.to(x.dtype)

    return out, s


# %%
@helion.kernel(static_shapes=True)
def se_block_bwd_dx(grad_out: Tensor, x: Tensor, w: Tensor, s: Tensor) -> Tensor:
    """
    Compute gradient for x.
    grad_x = 2 * grad_out * s + (2 * grad_out * x * s * (1 - s)) @ w.T

    Args:
        grad_out: Gradient w.r.t output [m, n]
        x: Input tensor [m, n]
        w: Weight matrix [n, n]
        s: sigmoid(x @ w) from forward pass [m, n]

    Returns:
        grad_x: Gradient w.r.t x [m, n]
    """
    m, n = x.size()

    grad_x = torch.empty([m, n], dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        # 2 * grad_out * s
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc += 2.0 * grad_out[tile_m, tile_n] * s[tile_m, tile_n]

        for tile_k in hl.tile(n):
            # 2 * grad_out * x * s * (1-s) for tile_k
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_k].to(torch.float32)
                * x[tile_m, tile_k].to(torch.float32)
                * s[tile_m, tile_k].to(torch.float32)
                * (1.0 - s[tile_m, tile_k].to(torch.float32))
            )
            # grad_to_w @ w.T[tile_k, tile_n] = grad_to_w @ w[tile_n, tile_k].T
            acc += grad_to_w @ w[tile_n, tile_k].to(torch.float32).T

        grad_x[tile_m, tile_n] = acc.to(x.dtype)

    return grad_x


# %%
@helion.kernel(static_shapes=True)
def se_block_bwd_dw(grad_out: Tensor, x: Tensor, s: Tensor) -> Tensor:
    """
    Compute gradient for w.
    grad_w = x.T @ (2 * grad_out * x * s * (1 - s))

    Args:
        grad_out: Gradient w.r.t output [m, n]
        x: Input tensor [m, n]
        s: sigmoid(x @ w) from forward pass [m, n]

    Returns:
        grad_w: Gradient w.r.t w [n, n]
    """
    m, n = x.size()

    grad_w = torch.zeros([n, n], dtype=torch.float32, device=x.device)

    for tile_n1, tile_n2 in hl.tile([n, n]):
        acc_w = hl.zeros([tile_n1, tile_n2], dtype=torch.float32)
        for tile_m in hl.tile(m):
            # 2 * grad_out * x * s * (1-s)
            grad_to_w = (
                2.0
                * grad_out[tile_m, tile_n2].to(torch.float32)
                * x[tile_m, tile_n2].to(torch.float32)
                * s[tile_m, tile_n2].to(torch.float32)
                * (1.0 - s[tile_m, tile_n2].to(torch.float32))
            )
            # x[tile_m, tile_n1].T @ grad_to_w[tile_m, tile_n2]
            acc_w += x[tile_m, tile_n1].to(torch.float32).T @ grad_to_w

        grad_w[tile_n1, tile_n2] = acc_w.to(x.dtype)

    return grad_w


# %%
# Reference Implementation
# --------------------
def se_block_pytorch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation se_block.

    Args:
        x, w: Input tensors

    Returns:
        tensor of 2 * x * sigmoid(x @ w)
    """
    return 2 * x * torch.sigmoid(x @ w)


# %%
# Autograd Function
# ------------------
class SEBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for se block."""
        out, s = se_block_fwd(x, w)
        ctx.save_for_backward(x, w, s)  # type: ignore[attr-defined]
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward pass for se block."""
        x, w, s = ctx.saved_tensors  # type: ignore[attr-defined]

        grad_x = se_block_bwd_dx(grad_out, x, w, s)
        grad_w = se_block_bwd_dw(grad_out, x, s)

        return grad_x, grad_w


def se_block(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    SE Block with autograd support.

    Args:
        x: Input tensor [m, n]
        w: Weight matrix [n, n]

    Returns:
        Output tensor [m, n]
    """
    return SEBlockFunction.apply(x, w)  # type: ignore[no-any-return]


def check(m: int, n: int) -> None:
    """
    Checks the correctness against PyTorch.
    Args:
        m (int): Number of rows in matrix x.
        n (int): Number of columns in matrix x.
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
    w = torch.randn([n, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
    for bwd in [True, False]:
        run_example(se_block, se_block_pytorch, (x, w), bwd=bwd)


# %%
def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(1024, 1024)


# %%
if __name__ == "__main__":
    main()
