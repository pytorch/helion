"""better-benchmark repro sum_sum_2cd99e5eb23e (genai/static RMSNormBackward).

RMSNorm backward for a 0-d bf16 upstream gradient ``g`` broadcast over the bf16
activations ``x`` ``[rows, hidden]``, with fp32 per-row ``rstd`` ``[rows, 1]``
and fp32 weight ``w`` ``[hidden]``: the input gradient
``g*w*rstd + sum(g*w*x, -1) * -0.5 * rstd**3 / hidden * (x * 2.0)`` in bf16 and
the weight gradient ``sum(g * (x * rstd), 0)`` in fp32.

Inductor emits one mix-order reduction,
``triton_per_fused_add_convert_element_type_div_expand_mul_pow_sum_0`` (the
other ``triton_`` launches in the dump are Inductor benchmarking candidate
split sizes): each program owns ``RSPLIT_SIZE = 32`` consecutive rows and loops
over them in row blocks, computing the per-row persistent reduction over
``hidden``, storing the input gradient, and accumulating the per-column
weight-gradient partial of its rows, which it stores to its row of an fp32
``[rows / 32, hidden]`` workspace. Inductor's wrapper then finishes the weight
gradient with an ATen ``workspace.sum(dim=0)``; the port does that column sum
in a second kernel instead of on the host.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split for this shape: 32 rows per workspace
# row (1152000 rows -> a [36000, 512] workspace).
RSPLIT_SIZE = 32


@helion.kernel(static_shapes=True)
def rmsnorm_bwd_grad_input_weight_partials(
    grad: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad: 0-d; weight: [hidden]; rstd: [splits, rows_per_split] and
    # x: [splits, rows_per_split, hidden], views of the [rows, 1] and
    # [rows, hidden] inputs.
    splits, rows_per_split, hidden = x.shape
    grad_input = torch.empty(
        [splits, rows_per_split, hidden], dtype=x.dtype, device=x.device
    )
    weight_partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        g = grad[()].to(torch.float32)
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            xf = x[tile_s, tile_r, :].to(torch.float32)
            r = rstd[tile_s, tile_r][:, :, None]
            gw = g * weight[None, None, :]
            row_dot = torch.sum(gw * xf, dim=-1, keepdim=True)
            scale = row_dot * -0.5 * r**3 / hidden
            grad_input[tile_s, tile_r, :] = (gw * r + scale * (xf * 2.0)).to(
                grad_input.dtype
            )
            acc = acc + torch.sum(g * (xf * r), dim=1)
        weight_partial[tile_s, :] = acc
    return grad_input, weight_partial


@helion.kernel(static_shapes=True)
def rmsnorm_bwd_weight_grad(weight_partial: torch.Tensor) -> torch.Tensor:
    splits, hidden = weight_partial.shape
    weight_grad = torch.empty(
        [hidden], dtype=torch.float32, device=weight_partial.device
    )
    for tile_h in hl.tile(hidden):
        weight_grad[tile_h] = torch.sum(weight_partial[:, tile_h], dim=0)
    return weight_grad


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, weight, rstd, x = inputs[:4]
    weight_grad_shape = inputs[6]  # _shape_param_2 = [hidden]
    rows, hidden = x.shape
    splits = rows // RSPLIT_SIZE
    grad_input, weight_partial = rmsnorm_bwd_grad_input_weight_partials(
        grad,
        weight,
        rstd.view(splits, RSPLIT_SIZE),
        x.view(splits, RSPLIT_SIZE, hidden),
    )
    weight_grad = rmsnorm_bwd_weight_grad(weight_partial)
    return grad_input.view(rows, hidden), weight_grad.view(weight_grad_shape)
