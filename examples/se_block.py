"""
Helion SE Block Example
============================
This example demonstrates a Helion kernel implementation of simplified SE Block (Squeeze and Excitation Net).
Basically it performs excitation on embedding, similar as Squeeze and Excitation Net as those used
in https://arxiv.org/abs/1709.01507, the idea is to enhance signal/noise ratio to preserve useful information.
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

    # Tile the output columns and the reduction: a whole-row ``x_tile @ w``
    # needs all of w (n * n elements) on chip at once, which exceeds shared
    # memory from n = 512 on.
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        # Match the input-dtype matmul result before applying sigmoid.
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        # Compute output: 2 * x * sigmoid, cast to input dtype
        x_tile = x[tile_m, tile_n].to(torch.float32)
        out[tile_m, tile_n] = (2.0 * x_tile * sigmoid_result).to(x.dtype)

    return out, s


# %%
@helion.kernel(static_shapes=True)
def se_block_bwd_grad_to_w(grad_out: Tensor, x: Tensor, s: Tensor) -> Tensor:
    """
    The matmul operand both gradients share:
    g = 2 * grad_out * x * s * (1 - s), formed in fp32 and rounded once to
    the input dtype (the reference backward's operand of ``g @ w.T`` and
    ``x.T @ g``).  Computing it once here, instead of inside each gradient's
    reduction loop, reads grad_out / x / s once rather than twice.

    Args:
        grad_out: Gradient w.r.t output [m, n]
        x: Input tensor [m, n]
        s: sigmoid(x @ w) from forward pass [m, n]

    Returns:
        g: [m, n] in x.dtype
    """
    m, n = x.size()

    g = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        g[tile_m, tile_n] = (
            2.0
            * grad_out[tile_m, tile_n].to(torch.float32)
            * x[tile_m, tile_n].to(torch.float32)
            * s[tile_m, tile_n].to(torch.float32)
            * (1.0 - s[tile_m, tile_n].to(torch.float32))
        ).to(x.dtype)

    return g


# %%
@helion.kernel(static_shapes=True)
def se_block_bwd_dx(grad_out: Tensor, g: Tensor, w: Tensor, s: Tensor) -> Tensor:
    """
    Compute gradient for x.
    grad_x = 2 * grad_out * s + g @ w.T

    ``g`` is the shared operand from :func:`se_block_bwd_grad_to_w` (already
    in the input dtype); ``w`` is read as is, and the products accumulate in
    fp32.  grad_x is written in the input dtype, so autograd needs no
    converting copy.

    Args:
        grad_out: Gradient w.r.t output [m, n]
        g: 2 * grad_out * x * s * (1 - s) [m, n], in the input dtype
        w: Weight matrix [n, n]
        s: sigmoid(x @ w) from forward pass [m, n]

    Returns:
        grad_x: Gradient w.r.t x [m, n], in g.dtype
    """
    m, n = g.size()

    grad_x = torch.empty([m, n], dtype=g.dtype, device=g.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            # g @ w.T[tile_k, tile_n] = g @ w[tile_n, tile_k].T
            acc = torch.addmm(acc, g[tile_m, tile_k], w[tile_n, tile_k].T)

        # + 2 * grad_out * s, added after the contraction so the matmul
        # pipeline starts from a zero accumulator
        grad_x[tile_m, tile_n] = (
            acc + 2.0 * grad_out[tile_m, tile_n] * s[tile_m, tile_n]
        ).to(g.dtype)

    return grad_x


# %%
@helion.kernel(static_shapes=True)
def se_block_bwd_dw(x: Tensor, g: Tensor) -> Tensor:
    """
    Compute gradient for w.
    grad_w = x.T @ g

    ``g`` is the shared operand from :func:`se_block_bwd_grad_to_w`; ``x.T``
    is read as is, and the products accumulate in fp32.  Every tile of
    grad_w is written by exactly one iteration, so it is allocated
    uninitialized in the input dtype (no memset, no converting copy in
    autograd).

    Args:
        x: Input tensor [m, n]
        g: 2 * grad_out * x * s * (1 - s) [m, n], in the input dtype

    Returns:
        grad_w: Gradient w.r.t w [n, n], in x.dtype
    """
    m, n = x.size()

    grad_w = torch.empty([n, n], dtype=x.dtype, device=x.device)

    for tile_n1, tile_n2 in hl.tile([n, n]):
        acc_w = hl.zeros([tile_n1, tile_n2], dtype=torch.float32)
        for tile_m in hl.tile(m):
            # x[tile_m, tile_n1].T @ g[tile_m, tile_n2]
            acc_w = torch.addmm(acc_w, x[tile_m, tile_n1].T, g[tile_m, tile_n2])

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

        g = se_block_bwd_grad_to_w(grad_out, x, s)
        grad_x = se_block_bwd_dx(grad_out, g, w, s)
        grad_w = se_block_bwd_dw(x, g)

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
