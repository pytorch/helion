#!/usr/bin/env python3
"""Standalone reproduction of Helion layer_norm backward symbolic mismatch."""

from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion
import helion.language as hl


@helion.kernel
def _layer_norm_fwd(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.size()
    assert weight.size(0) == n
    if bias is not None:
        assert bias.size(0) == n
    assert len(normalized_shape) == 1
    assert normalized_shape[0] == n

    out = torch.empty_like(x)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean_val = torch.sum(acc, dim=-1) / n
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        rstd_val = torch.rsqrt(var_val + eps)
        normalized = centered * rstd_val[:, None]
        if bias is not None:
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
        else:
            acc = normalized * (weight[:].to(torch.float32))
        out[tile_m, :] = acc.to(x.dtype)
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val
    return out, mean, rstd


@helion.kernel
def _layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    compute_bias_grad: hl.constexpr = True,  # type: ignore[valid-type]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    m, n = x.shape  # x: [m, n] -> m: int, n: int
    n = hl.specialize(n)  # n: int (specialized feature size)
    weight_size = hl.specialize(weight.size(0))  # weight_size: int (=n)

    grad_x = torch.empty_like(x)  # grad_x: [m, n]

    m_block = hl.register_block_size(m)  # m_block: int tile size along batch dim
    num_blocks = (m + m_block - 1) // m_block  # num_blocks: int (ceil(m / m_block))

    grad_weight_partial = x.new_empty((num_blocks, weight_size), dtype=torch.float32)  # grad_weight_partial: [num_blocks, n]
    if compute_bias_grad:
        grad_bias_partial = x.new_empty((num_blocks, weight_size), dtype=torch.float32)  # grad_bias_partial: [num_blocks, n]
    else:
        grad_bias_partial = None  # grad_bias_partial: None

    weight_f32 = weight[:].to(torch.float32)  # weight_f32: [n]

    for mb_cta in hl.tile(m, block_size=m_block):  # mb_cta: tile descriptor over m
        grad_w_block = torch.zeros(weight_size, dtype=torch.float32, device=x.device)  # grad_w_block: [n]
        if compute_bias_grad:
            grad_b_block = torch.zeros(weight_size, dtype=torch.float32, device=x.device)  # grad_b_block: [n]
        for mb in hl.tile(mb_cta.begin, mb_cta.end):  # mb: tile covering subset of batch indices
            x_m = x[mb, :].to(torch.float32)  # x_m: [tile_m, n]
            dy_m = grad_out[mb, :].to(torch.float32)  # dy_m: [tile_m, n]
            mean_m = mean[mb].to(torch.float32)  # mean_m: [tile_m]
            rstd_m = rstd[mb].to(torch.float32)  # rstd_m: [tile_m]

            x_hat = (x_m - mean_m[:, None]) * rstd_m[:, None]  # x_hat: [tile_m, n]

            grad_w_block += torch.sum(dy_m * x_hat, dim=0)  # dy_m * x_hat: [tile_m, n] -> grad_w_block: [n]
            if compute_bias_grad:
                grad_b_block += torch.sum(dy_m, dim=0)  # dy_m: [tile_m, n] -> grad_b_block: [n]

            d_xhat = dy_m * weight_f32[None, :]  # d_xhat: [tile_m, n]
            mean_d_xhat = torch.sum(d_xhat, dim=-1, keepdim=True) / n  # mean_d_xhat: [tile_m, 1]
            mean_d_xhat_xhat = (
                torch.sum(d_xhat * x_hat, dim=-1, keepdim=True) / n
            )  # mean_d_xhat_xhat: [tile_m, 1]
            dx = (
                d_xhat - mean_d_xhat - x_hat * mean_d_xhat_xhat
            ) * rstd_m[:, None]  # dx: [tile_m, n]
            grad_x[mb, :] = dx.to(x.dtype)  # grad_x[mb, :]: [tile_m, n]

        grad_weight_partial[mb_cta.id, :] = grad_w_block  # grad_weight_partial[mb_cta.id, :]: [n]
        if compute_bias_grad:
            grad_bias_partial[mb_cta.id, :] = grad_b_block  # grad_bias_partial[mb_cta.id, :]: [n]

    grad_weight = grad_weight_partial.sum(0).to(weight.dtype)  # grad_weight_partial.sum(0): [n] -> grad_weight: [n]
    if compute_bias_grad:
        grad_bias = grad_bias_partial.sum(0).to(weight.dtype)  # grad_bias_partial.sum(0): [n] -> grad_bias: [n]
    else:
        grad_bias = None  # grad_bias: None

    return grad_x, grad_weight, grad_bias  # return: grad_x [m, n], grad_weight [n], grad_bias [n] or None


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        normalized_shape: list[int],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        y, mean, rstd = _layer_norm_fwd(x, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)  # type: ignore[arg-type]
        return y

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, torch.Tensor | None, torch.Tensor | None, None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors  # type: ignore[attr-defined]
        compute_bias_grad = bias is not None
        grad_x, grad_weight, grad_bias = _layer_norm_bwd(
            grad_output, x, weight, mean, rstd, compute_bias_grad
        )
        return grad_x, None, grad_weight, grad_bias, None


def _layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormFunction.apply(x, normalized_shape, weight, bias, eps)


def main() -> None:
    torch.manual_seed(0)

    m, n = 4096, 5632
    device = "cuda"
    dtype = torch.float16

    x = torch.randn((m, n), device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn((n,), device=device, dtype=dtype, requires_grad=True)

    print(f"Running Helion layer_norm backward with shape {(m, n)}…")

    try:
        y = _layer_norm(x, [n], weight, bias, 1e-5)
        dy = torch.randn_like(y)
        y.backward(dy)
    except Exception as exc:  # noqa: BLE001
        print("Caught exception (expected repro):")
        raise
    else:
        print("No exception raised – bug not reproduced.")


if __name__ == "__main__":
    main()
