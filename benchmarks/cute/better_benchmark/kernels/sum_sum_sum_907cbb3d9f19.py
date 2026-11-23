"""better-benchmark repro sum_sum_sum_907cbb3d9f19 (hf/train/GoogleFnet).

FNet training backward: the backward of a LayerNorm whose upstream gradient is
the sum of two gradients of the same activations, and the scatter of the
LayerNorm input gradient into component 0 of an fp32
``[batch, seq, hidden, 2]`` tensor (it reads like the real part of the complex
gradient entering the backward of FNet's Fourier transform). With the fp32
gradients ``g`` ``[rows, hidden]`` (``rows = batch * seq``) and ``other``
``[batch, seq, hidden]``, the fp32 LayerNorm weight ``w`` ``[hidden]``, the
fp32 normalized input ``x_hat`` ``[batch, seq, hidden]``, an fp32 per-row
factor ``scale`` ``[batch, seq, 1]`` and the fp32 ``pair``
``[batch, seq, hidden, 2]``: ``add = other + g``, ``sum_1 = sum(add * w, -1)``,
``sum_2 = sum(add * w * x_hat, -1)``, the LayerNorm input gradient
``scale * ((add * w * 768 - sum_1) - x_hat * sum_2)`` (``768`` is the graph
literal), the LayerNorm weight and bias gradients
``sum(add * x_hat, (0, 1))`` and ``sum(add, (0, 1))``, and
``select_scatter(pair, input gradient, 3, 0)``; everything is fp32. Inductor
emits two kernels and finishes the two column sums with ATen reductions:

- ``triton_per_fused_add_mul_sub_sum_view_0``: a mix-order reduction (the
  ``triton_`` launches in the dump are Inductor's benchmarking candidates for
  the split size): each program owns ``RSPLIT_SIZE = 32`` consecutive rows and
  loops over them in row blocks, computing the two per-row persistent
  reductions over ``hidden``, storing the input gradient, and accumulating the
  per-column partials of ``add * x_hat`` and of ``add`` over its rows, which
  it stores to its rows of the two ``[rows / 32, hidden]`` halves of one fp32
  ``[2, rows / 32, hidden]`` workspace.
- Inductor's wrapper then sums each half of the workspace over its rows with
  an ATen ``workspace.sum(dim=0)``; the port does each of the two column sums
  with a launch of ``sum_gradient_partials`` instead of on the host.
- ``triton_poi_fused_1``: the ``select_scatter``: every element of the fp32
  ``[batch, seq, hidden, 2]`` output loads its element of ``pair`` and the
  input-gradient element of its ``[batch, seq, hidden]`` position and keeps
  the gradient in component 0 and ``pair`` in component 1. Inductor indexes
  the output as one flat dimension and recovers the component and the
  gradient element with ``i % 2`` and ``i // 2``; the port tiles the
  ``[rows * hidden, 2]`` view.

The split divides the rows evenly at the repro's one shape point, so the
mix-order kernel reads ``[splits, rows_per_split, hidden]`` views of the dense
inputs; for a row count the blocks do not divide, the view fails before the
first launch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split at shape point 356087b1: 32 rows per
# workspace row (16384 rows -> two [512, 768] workspace halves).
RSPLIT_SIZE = 32


@helion.kernel(static_shapes=True)
def layernorm_bwd_partials(
    other: torch.Tensor,
    grad: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # weight: [hidden]; other, grad, x_hat (fp32):
    # [splits, rows_per_split, hidden] and scale: [splits, rows_per_split],
    # views of the [batch, seq, hidden] / [rows, hidden] and [batch, seq, 1]
    # inputs. workspace[0] holds the weight-gradient partials and
    # workspace[1] the bias-gradient partials, as in Inductor's workspace.
    splits, rows_per_split, hidden = other.shape
    grad_input = torch.empty(
        [splits, rows_per_split, hidden], dtype=torch.float32, device=other.device
    )
    workspace = torch.empty(
        [2, splits, hidden], dtype=torch.float32, device=other.device
    )
    for tile_s in hl.tile(splits):
        weight_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        bias_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            g = other[tile_s, tile_r, :] + grad[tile_s, tile_r, :]
            xh = x_hat[tile_s, tile_r, :]
            gw = g * weight[None, None, :]
            row_sum = torch.sum(gw, dim=-1, keepdim=True)
            row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
            centered = gw * norm_size - row_sum
            grad_input[tile_s, tile_r, :] = scale[tile_s, tile_r][:, :, None] * (
                centered - xh * row_dot
            )
            weight_acc = weight_acc + torch.sum(g * xh, dim=1)
            bias_acc = bias_acc + torch.sum(g, dim=1)
        workspace[0, tile_s, :] = weight_acc
        workspace[1, tile_s, :] = bias_acc
    return grad_input, workspace


@helion.kernel(static_shapes=True)
def sum_gradient_partials(partials: torch.Tensor) -> torch.Tensor:
    # partials: [splits, hidden] fp32, one half of the mix-order workspace;
    # returns its per-column sum as fp32 [hidden].
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partials.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partials[:, tile_h], dim=0)
    return out


@helion.kernel(static_shapes=True)
def select_scatter_component(
    pair: torch.Tensor, grad_input: torch.Tensor, component: hl.constexpr
) -> torch.Tensor:
    # pair: [elements, 2] fp32 and grad_input: [elements] fp32, views of the
    # [batch, seq, hidden, 2] input and of the [batch, seq, hidden] gradient;
    # returns select_scatter(pair, grad_input, 1, component).
    elements, components = pair.shape
    out = torch.empty([elements, components], dtype=pair.dtype, device=pair.device)
    for tile_e, tile_c in hl.tile([elements, components]):
        scattered = (tile_c.index == component)[None, :]
        out[tile_e, tile_c] = torch.where(
            scattered, grad_input[tile_e][:, None], pair[tile_e, tile_c]
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, other, weight, x_hat, scale, pair = inputs[:6]
    batch, seq, hidden = inputs[6]  # _shape_param_0 = [batch, seq, hidden]
    rows = batch * seq
    splits = rows // RSPLIT_SIZE
    grad_input, workspace = layernorm_bwd_partials(
        other.view(splits, RSPLIT_SIZE, hidden),
        grad.view(splits, RSPLIT_SIZE, hidden),
        weight,
        x_hat.view(splits, RSPLIT_SIZE, hidden),
        scale.view(splits, RSPLIT_SIZE),
        768,
    )
    weight_grad = sum_gradient_partials(workspace[0])
    bias_grad = sum_gradient_partials(workspace[1])
    scattered = select_scatter_component(
        pair.view(rows * hidden, pair.size(3)), grad_input.view(rows * hidden), 0
    )
    return (
        grad_input.view(batch, seq, hidden),
        weight_grad,
        bias_grad,
        scattered.view(pair.shape),
    )
