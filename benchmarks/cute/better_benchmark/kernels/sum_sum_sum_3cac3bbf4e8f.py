"""better-benchmark repro sum_sum_sum_3cac3bbf4e8f (genai/static LayerNormBackward).

LayerNorm backward for a 0-d bf16 upstream gradient ``g`` expanded over the
bf16 input ``x`` ``[rows, 512]``, with the fp32 weight ``w`` ``[512]`` and
the saved fp32 per-row ``mean`` and ``rstd`` ``[rows, 1]``: with
``x_hat = (x - mean) * rstd``, ``sum_1 = sum(g * w, -1)`` and
``sum_2 = sum(g * w * x_hat, -1)``, the input gradient
``rstd / 512 * ((g * w * 512 - sum_1) - x_hat * sum_2)`` in bf16 and the
weight gradient ``sum(g * x_hat, 0)`` in fp32 (``512`` is the graph literal).
Inductor emits two kernels and finishes with an ATen reduction:

- ``triton_per_fused_convert_element_type_expand_mul_sum_0``: persistent
  reduction of ``sum_1`` over the expanded gradient for every row, stored to
  an fp32 ``[rows]`` buffer.
- ``triton_per_fused_convert_element_type_div_expand_mul_sub_sum_1``: a
  mix-order reduction (the other ``triton_`` launches in the dump are
  Inductor's benchmarking candidates for the split size): each program owns
  ``RSPLIT_SIZE = 32`` consecutive rows and loops over them in row blocks,
  computing the per-row persistent reduction ``sum_2``, storing the input
  gradient, and accumulating the per-column weight-gradient partial of its
  rows, which it stores to its row of an fp32 ``[rows / 32, 512]`` workspace.
- Inductor's wrapper then sums the workspace over its rows with an ATen
  ``workspace.sum(dim=0)``; the port does that column sum in a third kernel
  instead of on the host.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split for this shape: 32 rows per workspace
# row (1152000 rows -> a [36000, 512] workspace).
RSPLIT_SIZE = 32


@helion.kernel(static_shapes=True)
def layernorm_bwd_gw_row_sum(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # grad: the 0-d upstream gradient expanded to [rows, hidden] (stride 0);
    # weight: [hidden].
    rows, hidden = grad.shape
    row_sum = torch.empty([rows], dtype=torch.float32, device=weight.device)
    for tile_m in hl.tile(rows):
        gw = grad[tile_m, :].to(torch.float32) * weight[None, :]
        row_sum[tile_m] = torch.sum(gw, dim=-1)
    return row_sum


@helion.kernel(static_shapes=True)
def layernorm_bwd_grad_input_weight_partials(
    grad: torch.Tensor,
    weight: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    row_sum: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad: 0-d; weight: [hidden]; x: [splits, rows_per_split, hidden] and
    # mean, rstd, row_sum: [splits, rows_per_split], views of the
    # [rows, hidden] input and the [rows, 1] / [rows] row statistics.
    splits, rows_per_split, hidden = x.shape
    grad_input = torch.empty(
        [splits, rows_per_split, hidden], dtype=x.dtype, device=x.device
    )
    weight_partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        g = grad[()].to(torch.float32)
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            gw = g * weight[None, None, :]
            r = rstd[tile_s, tile_r][:, :, None]
            x_hat = (
                x[tile_s, tile_r, :].to(torch.float32)
                - mean[tile_s, tile_r][:, :, None]
            ) * r
            row_dot = torch.sum(gw * x_hat, dim=-1, keepdim=True)
            gw_centered = gw * norm_size - row_sum[tile_s, tile_r][:, :, None]
            y = (r / norm_size) * (gw_centered - x_hat * row_dot)
            grad_input[tile_s, tile_r, :] = y.to(grad_input.dtype)
            acc = acc + torch.sum(g * x_hat, dim=1)
        weight_partial[tile_s, :] = acc
    return grad_input, weight_partial


@helion.kernel(static_shapes=True)
def layernorm_bwd_weight_grad(weight_partial: torch.Tensor) -> torch.Tensor:
    splits, hidden = weight_partial.shape
    weight_grad = torch.empty(
        [hidden], dtype=torch.float32, device=weight_partial.device
    )
    for tile_h in hl.tile(hidden):
        weight_grad[tile_h] = torch.sum(weight_partial[:, tile_h], dim=0)
    return weight_grad


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    rstd, grad, weight, x, mean = inputs[:5]
    rows, hidden = inputs[5]  # _shape_param_0 = [rows, hidden], the expand size
    splits = rows // RSPLIT_SIZE
    row_sum = layernorm_bwd_gw_row_sum(grad.expand(rows, hidden), weight)
    grad_input, weight_partial = layernorm_bwd_grad_input_weight_partials(
        grad,
        weight,
        x.view(splits, RSPLIT_SIZE, hidden),
        mean.view(splits, RSPLIT_SIZE),
        rstd.view(splits, RSPLIT_SIZE),
        row_sum.view(splits, RSPLIT_SIZE),
        512,
    )
    weight_grad = layernorm_bwd_weight_grad(weight_partial)
    return grad_input.view(rows, hidden), weight_grad
