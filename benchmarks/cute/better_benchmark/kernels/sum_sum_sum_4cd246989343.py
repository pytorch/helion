"""better-benchmark repro sum_sum_sum_4cd246989343 (hf/train/GoogleFnet).

FNet training backward: the backward of a block's output LayerNorm, whose
upstream gradient is the sum of two gradients of the same activations,
followed by a dropout backward and the column sums of its result (the bias
gradient of the linear layer before the dropout). One of the two gradients
is component 0 of an fp32 ``[batch, seq, hidden, 2]`` tensor (it reads like
the real part of the complex gradient of FNet's Fourier transform), which the
graph selects as a strided view. With that strided fp32 gradient ``real``
and the fp32 gradient ``other`` ``[batch, seq, hidden]``, the fp32 LayerNorm
weight ``w`` ``[hidden]``, the fp32 normalized input ``x_hat``
``[batch, seq, hidden]``, an fp32 per-row factor ``scale``
``[batch, seq, 1]`` and the bool dropout mask ``keep``:
``add = other + real``, ``sum_1 = sum(add * w, -1)``,
``sum_2 = sum(add * w * x_hat, -1)``, the LayerNorm input gradient
``scale * ((add * w * 768 - sum_1) - x_hat * sum_2)`` (``768`` is the graph
literal), the dropout backward of that gradient,
``* (keep * 1.1111111111111112)``, as ``[rows, hidden]``
(``rows = batch * seq``) and its transposed view, the LayerNorm weight and
bias gradients ``sum(add * x_hat, (0, 1))`` and ``sum(add, (0, 1))``, and the
column sums of the dropout backward over the rows; everything is fp32.
Inductor emits four kernels in six launches:

- ``triton_red_fused_add_mul_select_sum_0``: the first level of the split
  column sums of the weight and bias gradients: a looped reduction recomputes
  ``add`` and writes the per-column sums of ``add * x_hat`` and of ``add``
  over each block of ``ROWS_PER_SPLIT = 128`` consecutive rows to two fp32
  ``[rows / 128, hidden]`` workspaces.
- ``triton_red_fused_add_mul_select_sum_1``, launched twice: a looped
  reduction of one workspace per column into an fp32 ``[hidden]`` output,
  the weight gradient and then the bias gradient.
- ``triton_per_fused_add_convert_element_type_mul_select_sub_sum_2``: a
  persistent reduction over ``hidden``: each row recomputes ``add``, takes
  the two row sums and stores the LayerNorm input gradient and the dropout
  backward, both fp32 ``[batch, seq, hidden]``.
- ``triton_red_fused_convert_element_type_mul_sum_view_3``: the first level
  of the split column sum of the dropout backward, the same blocks of 128
  rows, into an fp32 ``[rows / 128, hidden]`` workspace.
- ``triton_red_fused_add_mul_select_sum_1`` again: sums those partials per
  column into the fp32 ``[hidden]`` output.

The split divides the rows evenly at the repro's one shape point, so the
split reductions read ``[splits, rows_per_split, hidden]`` views of the
inputs and of the dropout backward and sum a whole dim of their views; for
a row count the blocks do not divide, the view fails before the first
launch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split of the three column sums at shape point 3847e61c: 128
# blocks of 128 rows (16384 rows).
ROWS_PER_SPLIT = 128


@helion.kernel(static_shapes=True)
def grad_column_partial_sums(
    other: torch.Tensor, real: torch.Tensor, x_hat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # other, x_hat (dense) and real (strided, component 0 of the
    # [..., hidden, 2] input): [splits, rows_per_split, hidden] fp32 views of
    # the [batch, seq, hidden] gradients; returns each block's per-column
    # sums of add * x_hat and of add as fp32 [splits, hidden].
    splits, rows_per_split, hidden = other.shape
    weight_partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=other.device
    )
    bias_partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=other.device
    )
    for tile_s, tile_h in hl.tile([splits, hidden]):
        g = other[tile_s, :, tile_h] + real[tile_s, :, tile_h]
        weight_partial[tile_s, tile_h] = torch.sum(g * x_hat[tile_s, :, tile_h], dim=1)
        bias_partial[tile_s, tile_h] = torch.sum(g, dim=1)
    return weight_partial, bias_partial


@helion.kernel(static_shapes=True)
def sum_column_partials(partials: torch.Tensor) -> torch.Tensor:
    # partials: [splits, hidden] fp32; returns their per-column sum as fp32
    # [hidden].
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partials.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partials[:, tile_h], dim=0)
    return out


@helion.kernel(static_shapes=True)
def layernorm_bwd_dropout(
    other: torch.Tensor,
    real: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    keep: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # weight: [hidden]; other, x_hat (fp32, dense), real (fp32, strided) and
    # keep (bool): [rows, hidden] and scale: [rows], views of the
    # [batch, seq, hidden] and [batch, seq, 1] inputs.
    rows, hidden = other.shape
    grad_input = torch.empty([rows, hidden], dtype=torch.float32, device=other.device)
    grad_dropout = torch.empty([rows, hidden], dtype=torch.float32, device=other.device)
    for tile_r in hl.tile(rows):
        g = other[tile_r, :] + real[tile_r, :]
        xh = x_hat[tile_r, :]
        gw = g * weight[None, :]
        row_sum = torch.sum(gw, dim=-1, keepdim=True)
        row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
        centered = gw * norm_size - row_sum
        total = scale[tile_r][:, None] * (centered - xh * row_dot)
        grad_input[tile_r, :] = total
        kept = keep[tile_r, :].to(torch.float32) * 1.1111111111111112
        grad_dropout[tile_r, :] = total * kept
    return grad_input, grad_dropout


@helion.kernel(static_shapes=True)
def dropout_column_partial_sums(grad_rows: torch.Tensor) -> torch.Tensor:
    # grad_rows: [splits, rows_per_split, hidden] fp32, a view of the
    # [rows, hidden] dropout backward; returns each block's per-column sum
    # as fp32 [splits, hidden].
    splits, rows_per_split, hidden = grad_rows.shape
    partials = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_rows.device
    )
    for tile_s, tile_h in hl.tile([splits, hidden]):
        partials[tile_s, tile_h] = torch.sum(grad_rows[tile_s, :, tile_h], dim=1)
    return partials


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    pair, other, weight, x_hat, scale, keep = inputs[:6]
    rows, hidden = inputs[6]  # _shape_param_0 = [batch * seq, hidden]
    bias_grad_shape = inputs[7]  # _shape_param_1 = [hidden]
    splits = rows // ROWS_PER_SPLIT
    weight_partial, bias_partial = grad_column_partial_sums(
        other.view(splits, ROWS_PER_SPLIT, hidden),
        pair.view(splits, ROWS_PER_SPLIT, hidden, 2).select(3, 0),
        x_hat.view(splits, ROWS_PER_SPLIT, hidden),
    )
    weight_grad = sum_column_partials(weight_partial)
    norm_bias_grad = sum_column_partials(bias_partial)
    grad_input, grad_dropout = layernorm_bwd_dropout(
        other.view(rows, hidden),
        pair.view(rows, hidden, 2).select(2, 0),
        weight,
        x_hat.view(rows, hidden),
        scale.view(rows),
        keep.view(rows, hidden),
        768,
    )
    partials = dropout_column_partial_sums(
        grad_dropout.view(splits, ROWS_PER_SPLIT, hidden)
    )
    linear_bias_grad = sum_column_partials(partials)
    return (
        grad_input.view(other.shape),
        weight_grad,
        norm_bias_grad,
        grad_dropout,
        grad_dropout.permute(1, 0),
        linear_bias_grad.view(bias_grad_shape),
    )
