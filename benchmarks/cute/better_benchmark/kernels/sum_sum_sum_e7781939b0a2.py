"""better-benchmark repro sum_sum_sum_e7781939b0a2 (hf GPT-2 training).

GPT2ForSequenceClassification and DistillGPT2 training backward: a LayerNorm
backward added to another gradient of the same activations (the graph reads
like a block's skip connection), followed by a dropout backward and the
column sums of its result. With the bf16 upstream gradient ``g``
``[rows, hidden]``, the fp32 LayerNorm weight ``w`` ``[hidden]``, the fp32
normalized input ``x_hat`` ``[batch, seq, hidden]``, an fp32 per-row factor
``scale`` ``[batch, seq, 1]``, the fp32 gradient ``residual``
``[batch, seq, hidden]`` and the bool dropout mask ``keep``:
``sum_1 = sum(g * w, -1)``, ``sum_2 = sum(g * w * x_hat, -1)``, the summed
gradient ``add = residual + scale * ((g * w * 768 - sum_1) - x_hat * sum_2)``
in fp32 (``768`` is the graph literal), the dropout backward
``add * (keep * 1.1111111111111112)`` in bf16 as ``[rows, hidden]``, the
LayerNorm weight and bias gradients ``sum(g * x_hat, (0, 1))`` and
``sum(g, (0, 1))`` in fp32, and the fp32 column sums of the dropout backward
over the rows. Inductor emits three kernels and finishes two of the sums
with ATen reductions:

- ``triton_per_fused_add_convert_element_type_mul_sub_sum_view_0``: a
  mix-order reduction (the other ``triton_`` launches in the dump are
  Inductor's benchmarking candidates for the split size): each program owns
  ``RSPLIT_SIZE = 16`` consecutive rows and loops over them in row blocks,
  computing the two per-row persistent reductions, storing ``add`` and the
  dropout backward, and accumulating the per-column partials of the weight
  and bias gradients over its rows, which it stores to its rows of the two
  ``[rows / 16, hidden]`` halves of one fp32 ``[2, rows / 16, hidden]``
  workspace.
- Inductor's wrapper then sums each half of the workspace over its rows
  with an ATen ``workspace.sum(dim=0)``; the port does each of the two
  column sums with a launch of ``sum_gradient_partials`` instead of on the
  host.
- ``triton_red_fused_convert_element_type_mul_sum_view_1``: the first level
  of a split column sum of the bf16 dropout backward: a looped reduction
  writes the per-column sum of each of 64 blocks of ``rows / 64`` (128) rows
  to an fp32 ``[64, hidden]`` workspace.
- ``triton_per_fused_convert_element_type_mul_sum_view_2``: a persistent
  reduction of those partials per column into the fp32 ``[hidden]`` output.

As in Inductor (no ``emulate_precision_casts``), the math stays in fp32
between loads and stores: the dropout backward multiplies the fp32 ``add``
by the fp32 scaled mask and rounds once to bf16 (the graph rounds ``add`` and
the scaled mask to bf16 first), and the graph's bf16 round trip of the last
column sum is not materialized.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split at shape point 5a443972: 16 rows per
# workspace row (8192 rows -> two [512, 768] workspace halves).
RSPLIT_SIZE = 16
# Inductor's split of the dropout-backward column sum at that shape point:
# 64 blocks of rows / 64 (128) rows.
COLUMN_SUM_SPLITS = 64


@helion.kernel(static_shapes=True)
def layernorm_bwd_residual_dropout_partials(
    grad: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
    keep: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # weight: [hidden]; grad (bf16), x_hat, residual (fp32) and keep (bool):
    # [splits, rows_per_split, hidden] and scale: [splits, rows_per_split],
    # views of the [rows, hidden] / [batch, seq, hidden] and [batch, seq, 1]
    # inputs. workspace[0] holds the weight-gradient partials and
    # workspace[1] the bias-gradient partials, as in Inductor's workspace.
    splits, rows_per_split, hidden = grad.shape
    grad_residual = torch.empty(
        [splits, rows_per_split, hidden], dtype=torch.float32, device=grad.device
    )
    grad_dropout = torch.empty(
        [splits, rows_per_split, hidden], dtype=grad.dtype, device=grad.device
    )
    workspace = torch.empty(
        [2, splits, hidden], dtype=torch.float32, device=grad.device
    )
    for tile_s in hl.tile(splits):
        weight_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        bias_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            g = grad[tile_s, tile_r, :].to(torch.float32)
            xh = x_hat[tile_s, tile_r, :]
            gw = g * weight[None, None, :]
            row_sum = torch.sum(gw, dim=-1, keepdim=True)
            row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
            centered = gw * norm_size - row_sum
            total = residual[tile_s, tile_r, :] + scale[tile_s, tile_r][:, :, None] * (
                centered - xh * row_dot
            )
            grad_residual[tile_s, tile_r, :] = total
            kept = keep[tile_s, tile_r, :].to(torch.float32) * 1.1111111111111112
            grad_dropout[tile_s, tile_r, :] = (total * kept).to(grad_dropout.dtype)
            weight_acc = weight_acc + torch.sum(g * xh, dim=1)
            bias_acc = bias_acc + torch.sum(g, dim=1)
        workspace[0, tile_s, :] = weight_acc
        workspace[1, tile_s, :] = bias_acc
    return grad_residual, grad_dropout, workspace


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
def column_partial_sums(grad_rows: torch.Tensor) -> torch.Tensor:
    # grad_rows: [splits, rows_per_split, hidden] bf16, a view of the
    # [rows, hidden] dropout backward; returns each block's per-column sum
    # as fp32 [splits, hidden].
    splits, rows_per_split, hidden = grad_rows.shape
    partials = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_rows.device
    )
    for tile_s, tile_h in hl.tile([splits, hidden]):
        partials[tile_s, tile_h] = torch.sum(
            grad_rows[tile_s, :, tile_h].to(torch.float32), dim=1
        )
    return partials


@helion.kernel(static_shapes=True)
def sum_column_partials(partials: torch.Tensor) -> torch.Tensor:
    # partials: [splits, hidden] fp32; returns their per-column sum as fp32
    # [hidden].
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partials.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partials[:, tile_h], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, weight, x_hat, scale, residual, keep = inputs[:6]
    residual_shape = inputs[6]  # _shape_param_0 = [batch, seq, hidden]
    rows, hidden = inputs[7]  # _shape_param_1 = [batch * seq, hidden]
    bias_grad_shape = inputs[8]  # _shape_param_2 = [hidden]
    splits = rows // RSPLIT_SIZE
    grad_residual, grad_dropout, workspace = layernorm_bwd_residual_dropout_partials(
        grad.view(splits, RSPLIT_SIZE, hidden),
        weight,
        x_hat.view(splits, RSPLIT_SIZE, hidden),
        residual.view(splits, RSPLIT_SIZE, hidden),
        scale.view(splits, RSPLIT_SIZE),
        keep.view(splits, RSPLIT_SIZE, hidden),
        768,
    )
    weight_grad = sum_gradient_partials(workspace[0])
    norm_bias_grad = sum_gradient_partials(workspace[1])
    grad_dropout = grad_dropout.view(rows, hidden)
    partials = column_partial_sums(
        grad_dropout.view(COLUMN_SUM_SPLITS, rows // COLUMN_SUM_SPLITS, hidden)
    )
    linear_bias_grad = sum_column_partials(partials)
    return (
        weight_grad,
        norm_bias_grad,
        grad_residual.view(residual_shape),
        grad_dropout,
        linear_bias_grad.view(bias_grad_shape),
    )
