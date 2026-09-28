"""
Helion squeeze and excitation net Example
============================
This example demonstrates a Helion kernel implementation of squeeze and excitation
net as those used in https://arxiv.org/abs/1709.01507.
"""

# %%
from __future__ import annotations

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def squeeze_and_excitation_net_fwd(
    x: Tensor, a: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Performs torch.mul(x, torch.sigmoid(torch.relu((x @ a)) @ b))
    Args:
        x: 2D tensor of shape [m, n].
        a: 2D tensor of shape [n, k].
        b: 2D tensor of shape [k, n].
    Returns:
        out: Resulting matrix of shape [m, n].
        c = torch.relu(x @ a) of shape [m, k].
        d = torch.sigmoid(c @ b) of shape [m, n].
    """
    m, n = x.size()
    k = a.size(1)

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    d = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # Compute c = relu(x @ a) for this tile_m
        for tile_k in hl.tile(k):
            partial_xa = x[tile_m, :] @ a[:, tile_k]
            c[tile_m, tile_k] = torch.relu(partial_xa)

        # Compute d = sigmoid(c @ b) and out = x * d for this tile_m
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, c[tile_m, tile_k], b[tile_k, tile_n])
            # Match the input-dtype matmul result before applying sigmoid.
            d[tile_m, tile_n] = torch.sigmoid(acc.to(x.dtype))
            out[tile_m, tile_n] = x[tile_m, tile_n] * d[tile_m, tile_n]

    return out, c, d


# %%
@helion.kernel(static_shapes=True)
def squeeze_and_excitation_net_bwd_dx(
    grad_out: Tensor, x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor
) -> Tensor:
    """
    Compute grad_x for the squeeze and excitation network.
    grad_x = grad_out * d + (grad_out * x * d * (1-d) @ b.T * (c>0)) @ a.T

    Materialize grad_cb and grad_c once, then use tiled matmuls for the two
    linear layers. This avoids recomputing the full grad_c reduction for every
    output tile. Barriers make each intermediate available to the next phase.
    """
    m, n = x.size()
    k = a.size(1)

    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for point_m, point_n in hl.tile((m, n)):
        grad_cb[point_m, point_n] = (
            grad_out[point_m, point_n]
            * x[point_m, point_n]
            * d[point_m, point_n]
            * (1.0 - d[point_m, point_n])
        )
    hl.barrier()

    for row_c, col_c in hl.tile((m, k)):
        acc_c = hl.zeros([row_c, col_c], dtype=torch.float32)
        for reduce_n in hl.tile(n):
            acc_c = torch.addmm(acc_c, grad_cb[row_c, reduce_n], b[col_c, reduce_n].T)
        grad_c[row_c, col_c] = acc_c.to(x.dtype) * (c[row_c, col_c] > 0)
    hl.barrier()

    for row_x, col_x in hl.tile((m, n)):
        acc_x = hl.zeros([row_x, col_x], dtype=torch.float32)
        for reduce_k in hl.tile(k):
            acc_x = torch.addmm(acc_x, grad_c[row_x, reduce_k], a[col_x, reduce_k].T)
        grad_x[row_x, col_x] = grad_out[row_x, col_x] * d[row_x, col_x] + acc_x.to(
            x.dtype
        )

    return grad_x


# %%
@helion.kernel(static_shapes=True)
def squeeze_and_excitation_net_bwd_da(
    grad_out: Tensor, x: Tensor, b: Tensor, c: Tensor, d: Tensor
) -> Tensor:
    """
    Compute grad_a for the squeeze and excitation network.
    grad_a = x.T @ (grad_out * x * d * (1-d) @ b.T * (c>0))

    Materialize the intermediate gradients so both matmuls have explicit tiled
    reductions, without a full-row matmul nested inside another reduction.
    """
    m, n = x.size()
    k = c.size(1)

    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_a = torch.empty([n, k], dtype=x.dtype, device=x.device)

    for point_m, point_n in hl.tile((m, n)):
        grad_cb[point_m, point_n] = (
            grad_out[point_m, point_n]
            * x[point_m, point_n]
            * d[point_m, point_n]
            * (1.0 - d[point_m, point_n])
        )
    hl.barrier()

    for row_c, col_c in hl.tile((m, k)):
        acc_c = hl.zeros([row_c, col_c], dtype=torch.float32)
        for reduce_n in hl.tile(n):
            acc_c = torch.addmm(acc_c, grad_cb[row_c, reduce_n], b[col_c, reduce_n].T)
        grad_c[row_c, col_c] = acc_c.to(x.dtype) * (c[row_c, col_c] > 0)
    hl.barrier()

    for row_a, col_a in hl.tile((n, k)):
        acc_a = hl.zeros([row_a, col_a], dtype=torch.float32)
        for reduce_m in hl.tile(m):
            acc_a = torch.addmm(acc_a, x[reduce_m, row_a].T, grad_c[reduce_m, col_a])
        grad_a[row_a, col_a] = acc_a

    return grad_a


# %%
@helion.kernel(static_shapes=True)
def squeeze_and_excitation_net_bwd_db(
    grad_out: Tensor, x: Tensor, d: Tensor, c: Tensor
) -> Tensor:
    """
    Compute grad_b by fusing grad_d computation inline.
    grad_b = c.T @ (grad_out * x * d * (1 - d))
    """
    m, n = grad_out.size()
    k = c.size(1)
    grad_b = torch.empty([k, n], dtype=grad_out.dtype, device=grad_out.device)

    for tile_k, tile_n in hl.tile([k, n]):
        acc = hl.zeros([tile_k, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_d = (
                grad_out[tile_m, tile_n]
                * x[tile_m, tile_n]
                * d[tile_m, tile_n]
                * (1.0 - d[tile_m, tile_n])
            )
            acc = torch.addmm(acc, c[tile_m, tile_k].T, grad_d)
        grad_b[tile_k, tile_n] = acc

    return grad_b


# %%
# Reference Implementation
# --------------------
def squeeze_and_excitation_net_pytorch(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch reference implementation of squeeze_and_excitation_net.

    Args:
        x, a, b: Input tensors

    Returns:
        tensor of torch.mul(x, torch.sigmoid(torch.relu((x @ a)) @ b))
    """
    return torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b))


# %%
# Autograd Function
# ------------------
class SqueezeAndExcitationNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for squeeze and excitation network."""
        out, c, d = squeeze_and_excitation_net_fwd(x, a, b)
        ctx.save_for_backward(x, a, b, c, d)  # type: ignore[attr-defined]
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass for squeeze and excitation network."""
        x, a, b, c, d = ctx.saved_tensors  # type: ignore[attr-defined]

        grad_x = squeeze_and_excitation_net_bwd_dx(grad_out, x, a, b, c, d)
        grad_a = squeeze_and_excitation_net_bwd_da(grad_out, x, b, c, d)
        grad_b = squeeze_and_excitation_net_bwd_db(grad_out, x, d, c)
        return grad_x, grad_a, grad_b


def squeeze_and_excitation_net(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Squeeze and excitation network with autograd support.

    Args:
        x: Input tensor [m, n]
        a: Weight matrix [n, k]
        b: Weight matrix [k, n]

    Returns:
        Output tensor [m, n]
    """
    return SqueezeAndExcitationNetFunction.apply(x, a, b)  # type: ignore[no-any-return]


def check(m: int, k: int, n: int) -> None:
    """
    Checks the correctness against PyTorch.
    Args:
        m (int): Number of rows in matrix x.
        n (int): Number of columns in matrix x.
        k (int): Number of columns in matrix a.
    """
    x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
    a = torch.randn([n, k], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
    b = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
    for bwd in [True, False]:
        run_example(
            squeeze_and_excitation_net,
            squeeze_and_excitation_net_pytorch,
            (x, a, b),
            bwd=bwd,
        )


# %%
def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
