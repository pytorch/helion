"""
Helion squeeze and excitation net Example
============================
This example demonstrates a Helion kernel implementation of squeeze and excitation
net as those used in https://arxiv.org/abs/1709.01507.

All example tensors use ``torch.float32`` on ``DEVICE`` (see ``_TENSOR_DTYPE``).
"""

# %%
from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# Example runs end-to-end in float32 (inputs, weights, outputs, saved backward tensors).
_TENSOR_DTYPE = torch.float32

# Baseline tolerances for fp32 matmul+fused ops vs PyTorch eager (also used for autotune).
_AUTOTUNE_BASELINE_RTOL = 1e-2
_AUTOTUNE_BASELINE_ATOL = 1e-2


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, tuple[int, ...]]:
    """Return (max_abs, max_rel, argmax_abs_index) for two same-shape tensors."""
    a32 = a.detach().to(torch.float32)
    b32 = b.detach().to(torch.float32)
    diff = (a32 - b32).abs()
    flat = diff.reshape(-1)
    flat_idx = int(flat.argmax().item())
    max_abs = float(flat[flat_idx].item())
    denom = b32.abs().clamp_min(1e-12).reshape(-1)[flat_idx]
    max_rel = float((max_abs / float(denom.item())))

    idx = flat_idx
    shape = diff.shape
    unraveled: list[int] = []
    for dim in reversed(shape):
        unraveled.append(idx % dim)
        idx //= dim
    return max_abs, max_rel, tuple(reversed(unraveled))


def _report_close(
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    got32 = got.detach().to(torch.float32)
    ref32 = ref.detach().to(torch.float32)
    diff = (got32 - ref32).abs()
    tol = atol + rtol * ref32.abs()
    mism = int((diff > tol).sum().item())
    max_abs, max_rel, where = _max_diff(got, ref)
    print(
        f"[se debug] {name}: mismatched={mism}/{got.numel()} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} at {where} "
        f"(atol={atol}, rtol={rtol})",
        file=sys.stderr,
        flush=True,
    )


def squeeze_and_excitation_net_fwd_reference(
    x: Tensor, a: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """PyTorch reference for :func:`squeeze_and_excitation_net_fwd` (autotune baseline)."""
    c = torch.relu(torch.matmul(x.to(torch.float32), a.to(torch.float32)))
    d = torch.sigmoid(torch.matmul(c, b.to(torch.float32)))
    out = x.to(torch.float32) * d
    dt = x.dtype
    # Keep intermediates in fp32 so backward uses stable masks/derivatives.
    return out.to(dt), c, d


def squeeze_and_excitation_net_bwd_dx_reference(
    grad_out: Tensor,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """PyTorch reference matching ``squeeze_and_excitation_net_bwd_dx``."""
    g = grad_out.to(torch.float32)
    xf = x.to(torch.float32)
    grad_to_d = g * xf * d.to(torch.float32) * (1.0 - d.to(torch.float32))
    grad_to_c = torch.matmul(grad_to_d, b.transpose(0, 1).to(torch.float32))
    grad_c = grad_to_c * (c.to(torch.float32) > 0)
    return (g * d.to(torch.float32) + torch.matmul(grad_c, a.transpose(0, 1).to(torch.float32))).to(
        x.dtype
    )


def squeeze_and_excitation_net_bwd_da_reference(
    grad_out: Tensor,
    x: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """PyTorch reference matching ``squeeze_and_excitation_net_bwd_da``."""
    grad_to_cb = (
        grad_out.to(torch.float32)
        * x.to(torch.float32)
        * d.to(torch.float32)
        * (1.0 - d.to(torch.float32))
    )
    grad_to_c = torch.matmul(grad_to_cb, b.transpose(0, 1).to(torch.float32)) * (
        c.to(torch.float32) > 0
    )
    return torch.matmul(x.transpose(0, 1).to(torch.float32), grad_to_c).to(x.dtype)


def squeeze_and_excitation_net_bwd_db_reference(
    grad_out: Tensor,
    x: Tensor,
    d: Tensor,
    c: Tensor,
) -> Tensor:
    """PyTorch reference matching ``squeeze_and_excitation_net_bwd_db``."""
    grad_d = (
        grad_out.to(torch.float32)
        * x.to(torch.float32)
        * d.to(torch.float32)
        * (1.0 - d.to(torch.float32))
    )
    return torch.matmul(c.transpose(0, 1).to(torch.float32), grad_d).to(grad_out.dtype)


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=squeeze_and_excitation_net_fwd_reference,
    autotune_baseline_rtol=_AUTOTUNE_BASELINE_RTOL,
    autotune_baseline_atol=_AUTOTUNE_BASELINE_ATOL,
)
def squeeze_and_excitation_net_fwd(
    x: Tensor,
    a: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Performs torch.mul(x, torch.sigmoid(torch.relu((x @ a)) @ b))

    Tile extents are chosen by autotuning (no fixed ``block_size`` on ``hl.tile``).

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

    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    c = torch.empty([m, k], dtype=torch.float32, device=x.device)
    d = torch.empty([m, n], dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        for tile_k in hl.tile(k):
            acc_k = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            for tile_n in hl.tile(n):
                acc_k = acc_k + torch.matmul(x[tile_m, tile_n], a[tile_n, tile_k])
            c[tile_m, tile_k] = torch.relu(acc_k)

        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, c[tile_m, tile_k], b[tile_k, tile_n])
            d[tile_m, tile_n] = torch.sigmoid(acc)
            out[tile_m, tile_n] = x[tile_m, tile_n] * d[tile_m, tile_n]

    return out, c, d


# %%
@helion.kernel(
    static_shapes=True,
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=squeeze_and_excitation_net_bwd_dx_reference,
    autotune_baseline_rtol=_AUTOTUNE_BASELINE_RTOL,
    autotune_baseline_atol=_AUTOTUNE_BASELINE_ATOL,
)
def squeeze_and_excitation_net_bwd_dx(
    grad_out: Tensor,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """
    Compute grad_x for the squeeze and excitation network.
    grad_x = grad_out * d + (grad_out * x * d * (1-d) @ b.T * (c>0)) @ a.T

    The reduction over ``n`` for ``grad_to_d @ b.T`` uses an inner ``tile_nj`` loop.
    """
    m, n = x.size()
    k = a.size(1)

    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc += grad_out[tile_m, tile_n] * d[tile_m, tile_n]

        for tile_k in hl.tile(k):
            grad_to_c = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            for tile_nj in hl.tile(n):
                g = (
                    grad_out[tile_m, tile_nj]
                    * x[tile_m, tile_nj]
                    * d[tile_m, tile_nj]
                    * (1.0 - d[tile_m, tile_nj])
                )
                grad_to_c = grad_to_c + g @ b[tile_k, tile_nj].T
            grad_c_masked = grad_to_c * (c[tile_m, tile_k] > 0)
            acc = torch.addmm(acc, grad_c_masked, a[tile_n, tile_k].T)

        grad_x[tile_m, tile_n] = acc

    return grad_x


# %%
@helion.kernel(
    static_shapes=True,
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=squeeze_and_excitation_net_bwd_da_reference,
    autotune_baseline_rtol=_AUTOTUNE_BASELINE_RTOL,
    autotune_baseline_atol=_AUTOTUNE_BASELINE_ATOL,
)
def squeeze_and_excitation_net_bwd_da(
    grad_out: Tensor,
    x: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
) -> Tensor:
    """
    Compute grad_a for the squeeze and excitation network.
    grad_a = x.T @ (grad_out * x * d * (1-d) @ b.T * (c>0))

    ``grad_to_cb @ b[tile_k].T`` is accumulated along ``n`` in an inner loop.
    """
    m, n = x.size()
    k = c.size(1)

    grad_a = torch.empty([n, k], dtype=x.dtype, device=x.device)

    for tile_n, tile_k in hl.tile([n, k]):
        acc_a = hl.zeros([tile_n, tile_k], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_to_c = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            for tile_nj in hl.tile(n):
                grad_to_cb = (
                    grad_out[tile_m, tile_nj]
                    * x[tile_m, tile_nj]
                    * d[tile_m, tile_nj]
                    * (1.0 - d[tile_m, tile_nj])
                )
                grad_to_c = grad_to_c + grad_to_cb @ b[tile_k, tile_nj].T
            grad_through_relu = grad_to_c * (c[tile_m, tile_k] > 0)
            acc_a = torch.addmm(acc_a, x[tile_m, tile_n].T, grad_through_relu)
        grad_a[tile_n, tile_k] = acc_a

    return grad_a


# %%
@helion.kernel(
    static_shapes=True,
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=squeeze_and_excitation_net_bwd_db_reference,
    autotune_baseline_rtol=_AUTOTUNE_BASELINE_RTOL,
    autotune_baseline_atol=_AUTOTUNE_BASELINE_ATOL,
)
def squeeze_and_excitation_net_bwd_db(
    grad_out: Tensor,
    x: Tensor,
    d: Tensor,
    c: Tensor,
) -> Tensor:
    """
    ``grad_b = c.T @ (grad_out * x * d * (1 - d))`` with ``c = relu(x @ a)``.

    Helion requires at least one ``hl.tile``/``hl.grid``; accumulation over ``m`` is
    split across ``tile_m`` (same math as one ``matmul``, possibly different fp32 order).
    """
    m, n = grad_out.size()
    k = c.size(1)

    grad_b = torch.empty([k, n], dtype=grad_out.dtype, device=grad_out.device)

    for tile_k, tile_n in hl.tile([k, n]):
        acc_b = hl.zeros([tile_k, tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            grad_d = (
                grad_out[tile_m, tile_n]
                * x[tile_m, tile_n]
                * d[tile_m, tile_n]
                * (1.0 - d[tile_m, tile_n])
            )
            acc_b = torch.addmm(
                acc_b,
                c[tile_m, tile_k].transpose(0, 1),
                grad_d,
            )
        grad_b[tile_k, tile_n] = acc_b

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
    x = torch.randn([m, n], device=DEVICE, dtype=_TENSOR_DTYPE, requires_grad=True)
    a = torch.randn([n, k], device=DEVICE, dtype=_TENSOR_DTYPE, requires_grad=True)
    b = torch.randn([k, n], device=DEVICE, dtype=_TENSOR_DTYPE, requires_grad=True)

    for bwd in [True, False]:
        run_example(
            squeeze_and_excitation_net,
            squeeze_and_excitation_net_pytorch,
            (x, a, b),
            bwd=bwd,
            rtol=_AUTOTUNE_BASELINE_RTOL,
            atol=_AUTOTUNE_BASELINE_ATOL,
        )


# %%
def main() -> None:
    """
    Main function to run correctness checks.
    """
    check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    import time
    time_st = time.time()
    main()
    print(f"time cost: {time.time() - time_st}")
