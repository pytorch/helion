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
            d[tile_m, tile_n] = torch.sigmoid(acc)
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

    The computation is structured to properly accumulate over the k dimension:
    1. First term: grad_out * d (element-wise, no reduction)
    2. Second term: chain rule through d->c->x path
       - For each output position (m, n), accumulate over k dimension
       - grad_c[m,k] = (grad_out * x * d * (1-d))[m,:] @ b[k,:].T * (c[m,k] > 0)
       - grad_x[m,n] += grad_c[m,k] @ a[n,k].T
    """
    m, n = x.size()
    k = a.size(1)

    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)

    # Compute grad_x: grad_out * d + second_term where second_term accumulates over k
    for tile_m, tile_n in hl.tile([m, n]):
        # First term: grad_out * d (element-wise)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc += grad_out[tile_m, tile_n] * d[tile_m, tile_n]

        # Second term: accumulate gradient chain over k dimension
        for tile_k in hl.tile(k):
            # Compute grad_to_d for the full row: shape [tile_m, n]
            grad_to_d = (
                grad_out[tile_m, :] * x[tile_m, :] * d[tile_m, :] * (1.0 - d[tile_m, :])
            )

            # Backprop through (c @ b): grad_c = grad_to_d @ b.T
            # [tile_m, n] @ [n, tile_k] = [tile_m, tile_k]
            grad_to_c = grad_to_d @ b[tile_k, :].T

            # Apply ReLU mask: shape [tile_m, tile_k]
            grad_c_masked = grad_to_c * (c[tile_m, tile_k] > 0)

            # Backprop through (x @ a): grad_x_contribution = grad_c_masked @ a.T
            # [tile_m, tile_k] @ [tile_k, tile_n] = [tile_m, tile_n]
            acc = torch.addmm(acc, grad_c_masked, a[tile_n, tile_k].T)

        grad_x[tile_m, tile_n] = acc

    return grad_x


# %%
@helion.kernel(static_shapes=True)
def squeeze_and_excitation_net_bwd_da(
    grad_out: Tensor, x: Tensor, b: Tensor, c: Tensor, d: Tensor
) -> Tensor:
    """
    Compute grad_a for the squeeze and excitation network.
    grad_a = x.T @ (grad_out * x * d * (1-d) @ b.T * (c>0))
    """
    m, n = x.size()
    k = c.size(1)

    grad_a = torch.empty([n, k], dtype=x.dtype, device=x.device)

    # Compute grad_a: x.T @ grad_c
    for tile_n, tile_k in hl.tile([n, k]):
        acc_a = hl.zeros([tile_n, tile_k], dtype=torch.float32)
        for tile_m in hl.tile(m):
            # Backprop through sigmoid: need full row for matmul with b.T
            grad_to_d = grad_out[tile_m, :] * x[tile_m, :]
            grad_to_cb = grad_to_d * d[tile_m, :] * (1.0 - d[tile_m, :])
            # Backprop through c @ b: [tile_m, n] @ [n, tile_k] = [tile_m, tile_k]
            grad_to_c = grad_to_cb @ b[tile_k, :].T
            # Backprop through relu
            grad_through_relu = grad_to_c * (c[tile_m, tile_k] > 0)
            # Accumulate x.T @ grad_c: [tile_n, tile_m] @ [tile_m, tile_k] = [tile_n, tile_k]
            acc_a = torch.addmm(acc_a, x[tile_m, tile_n].T, grad_through_relu)
        grad_a[tile_n, tile_k] = acc_a

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
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
    a = torch.randn([n, k], device=DEVICE, dtype=torch.float16, requires_grad=True)
    b = torch.randn([k, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
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
