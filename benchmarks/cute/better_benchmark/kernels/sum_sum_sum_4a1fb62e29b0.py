"""better-benchmark repro sum_sum_sum_4a1fb62e29b0 (hf masked-LM / causal-LM training: BERT, LayoutLM, RoBERTa, DeBERTa-v2, DistilBERT, ELECTRA).

A transformer block's backward from a LayerNorm output back through the
LayerNorm and the dropout of the sublayer before it. The upstream gradient
``g`` of the LayerNorm output is the sum of four contributions, an fp32
``[batch, seq, hidden]`` one and three bf16 ``[rows, hidden]`` ones
(``rows = batch * seq``). With the fp32 LayerNorm weight ``w`` ``[hidden]``,
the saved fp32 normalized input ``x_hat`` ``[batch, seq, hidden]``, the fp32
per-row ``scale`` ``[batch, seq, 1]`` and the dropout keep mask:

- ``sum_1 = sum(g * w, -1)`` and ``sum_2 = sum(g * w * x_hat, -1)``,
- the LayerNorm input gradient
  ``grad_in = scale * ((g * w * 1536 - sum_1) - x_hat * sum_2)`` in fp32
  (``1536`` is the graph literal, the hidden size of the captured
  DeBERTa-v2, at every shape point),
- ``sum(g * x_hat, (0, 1))`` and ``sum(g, (0, 1))`` in fp32 (the LayerNorm
  weight and bias gradients),
- the dropout backward ``grad_in * (keep * 1.1111111111111112)`` in bf16 as
  ``[rows, hidden]``, and its transposed view,
- the fp32 column sum of that bf16 tensor over the rows (the bias gradient of
  the linear layer before the dropout).

Inductor emits five kernels and finishes two of the sums with ATen
reductions:

- ``triton_poi_fused_add_convert_element_type_mul_view_0``: pointwise
  ``g * w`` into an fp32 ``[batch, seq, hidden]`` buffer (four loads of the
  gradient contributions per element, the weight indexed by
  ``index % hidden``).
- ``triton_per_fused_add_convert_element_type_mul_sum_view_1``: a mix-order
  reduction (the ``triton_`` launches in the dump are Inductor's benchmarking
  candidates for the split size): each program owns ``RSPLIT_SIZE = 32``
  consecutive rows and loops over them in row blocks, computing the two
  per-row persistent reductions of the ``g * w`` buffer over ``hidden`` into
  two fp32 ``[rows]`` buffers, and, from ``g`` recomputed from its four
  contributions, accumulating the per-column partials of ``g * x_hat`` and
  ``g`` over its rows, which it stores to its rows of an fp32
  ``[2, rows / 32, hidden]`` workspace (two ``[rows / 32, hidden]`` buffers
  in the port).
- Inductor's wrapper then sums each half of the workspace over its rows with
  an ATen ``workspace.sum(dim=0)``; the port does these two column sums with
  two launches of ``sum_gradient_partials`` instead of on the host.
- ``triton_poi_fused_convert_element_type_mul_sub_2``: pointwise
  ``grad_in`` from the ``g * w`` buffer, the two per-row sums, ``x_hat`` and
  ``scale``, written in place over the ``g * w`` buffer, plus the bf16
  dropout backward.
- ``triton_red_fused_convert_element_type_mul_sum_view_3``: the first level
  of a split column sum of the bf16 dropout backward: a looped reduction
  writes the per-column sum of each of 128 blocks of ``rows / 128`` (128)
  rows to an fp32 ``[128, hidden]`` workspace.
- ``triton_red_fused_convert_element_type_mul_sum_view_4``: sums those
  partials per column into the fp32 ``[hidden]`` output.

As in Inductor (no ``emulate_precision_casts``), the math stays in fp32
between loads and stores: the dropout backward multiplies the fp32
``grad_in`` by the fp32 scaled mask and rounds once to bf16 (the graph rounds
``grad_in`` and the scaled mask to bf16 first), and the graph's bf16 round
trip of the last column sum is not materialized.

Both splits divide the rows evenly at every shape point of the repro, so the
reductions read ``[splits, rows_per_split, hidden]`` views of the dense
buffers, and the split column sum sums a whole dim of its view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split at shape point c1e38d67: 32 rows per
# workspace row (16384 rows -> two [512, 768] workspace halves).
RSPLIT_SIZE = 32
# Inductor's split of the dropout-backward column sum at that shape point:
# 128 blocks of rows / 128 (128) rows.
COLUMN_SUM_SPLITS = 128


@helion.kernel(static_shapes=True)
def summed_grad_times_weight(
    grad_fp32: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_c: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    # grad_fp32: [rows, hidden] fp32, a view of the [batch, seq, hidden]
    # input; grad_a, grad_b, grad_c: [rows, hidden] bf16; weight: [hidden].
    rows, hidden = grad_a.shape
    out = torch.empty([rows, hidden], dtype=torch.float32, device=grad_a.device)
    for tile_r, tile_h in hl.tile([rows, hidden]):
        g = grad_fp32[tile_r, tile_h] + grad_a[tile_r, tile_h].to(torch.float32)
        g = g + grad_b[tile_r, tile_h].to(torch.float32)
        g = g + grad_c[tile_r, tile_h].to(torch.float32)
        out[tile_r, tile_h] = g * weight[tile_h]
    return out


@helion.kernel(static_shapes=True)
def layernorm_bwd_row_sums_partials(
    grad_weighted: torch.Tensor,
    x_hat: torch.Tensor,
    grad_fp32: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # grad_weighted, x_hat, grad_fp32 (fp32) and grad_a, grad_b, grad_c
    # (bf16): [splits, rows_per_split, hidden] views of the
    # [batch, seq, hidden] / [rows, hidden] buffers.
    splits, rows_per_split, hidden = grad_weighted.shape
    row_sum = torch.empty(
        [splits, rows_per_split], dtype=torch.float32, device=grad_weighted.device
    )
    row_dot = torch.empty(
        [splits, rows_per_split], dtype=torch.float32, device=grad_weighted.device
    )
    weight_partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_weighted.device
    )
    bias_partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_weighted.device
    )
    for tile_s in hl.tile(splits):
        weight_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        bias_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            gw = grad_weighted[tile_s, tile_r, :]
            xh = x_hat[tile_s, tile_r, :]
            row_sum[tile_s, tile_r] = torch.sum(gw, dim=-1)
            row_dot[tile_s, tile_r] = torch.sum(gw * xh, dim=-1)
            g = grad_fp32[tile_s, tile_r, :] + grad_a[tile_s, tile_r, :].to(
                torch.float32
            )
            g = g + grad_b[tile_s, tile_r, :].to(torch.float32)
            g = g + grad_c[tile_s, tile_r, :].to(torch.float32)
            weight_acc = weight_acc + torch.sum(g * xh, dim=1)
            bias_acc = bias_acc + torch.sum(g, dim=1)
        weight_partial[tile_s, :] = weight_acc
        bias_partial[tile_s, :] = bias_acc
    return row_sum, row_dot, weight_partial, bias_partial


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
def layernorm_bwd_input_grad_dropout(
    grad_weighted: torch.Tensor,
    scale: torch.Tensor,
    row_sum: torch.Tensor,
    x_hat: torch.Tensor,
    row_dot: torch.Tensor,
    keep: torch.Tensor,
    norm_size: hl.constexpr,
    keep_scale: hl.constexpr,
) -> torch.Tensor:
    # grad_weighted (fp32, overwritten with the LayerNorm input gradient),
    # x_hat (fp32) and keep (bool): [rows, hidden]; scale, row_sum, row_dot:
    # [rows] fp32. Returns the dropout backward as bf16 [rows, hidden].
    rows, hidden = grad_weighted.shape
    grad_dropout = torch.empty(
        [rows, hidden], dtype=torch.bfloat16, device=grad_weighted.device
    )
    for tile_r, tile_h in hl.tile([rows, hidden]):
        centered = grad_weighted[tile_r, tile_h] * norm_size - row_sum[tile_r][:, None]
        grad_in = scale[tile_r][:, None] * (
            centered - x_hat[tile_r, tile_h] * row_dot[tile_r][:, None]
        )
        kept = keep[tile_r, tile_h].to(torch.float32) * keep_scale
        grad_weighted[tile_r, tile_h] = grad_in
        grad_dropout[tile_r, tile_h] = (grad_in * kept).to(grad_dropout.dtype)
    return grad_dropout


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
    # [hidden]. The graph rounds this sum to bf16 and back; Inductor keeps it
    # in fp32.
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partials.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partials[:, tile_h], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_a, grad_fp32, grad_b, grad_c, weight, x_hat, scale, keep = inputs[:8]
    batch, seq, hidden = inputs[8]  # _shape_param_0 = [batch, seq, hidden]
    rows, _ = inputs[11]  # _shape_param_3 = [batch * seq, hidden]
    bias_grad_shape = inputs[12]  # _shape_param_4 = [hidden]
    splits = rows // RSPLIT_SIZE
    grad_weighted = summed_grad_times_weight(
        grad_fp32.view(rows, hidden), grad_a, grad_b, grad_c, weight
    )
    row_sum, row_dot, weight_partial, bias_partial = layernorm_bwd_row_sums_partials(
        grad_weighted.view(splits, RSPLIT_SIZE, hidden),
        x_hat.view(splits, RSPLIT_SIZE, hidden),
        grad_fp32.view(splits, RSPLIT_SIZE, hidden),
        grad_a.view(splits, RSPLIT_SIZE, hidden),
        grad_b.view(splits, RSPLIT_SIZE, hidden),
        grad_c.view(splits, RSPLIT_SIZE, hidden),
    )
    weight_grad = sum_gradient_partials(weight_partial)
    norm_bias_grad = sum_gradient_partials(bias_partial)
    grad_dropout = layernorm_bwd_input_grad_dropout(
        grad_weighted,
        scale.view(rows),
        row_sum.view(rows),
        x_hat.view(rows, hidden),
        row_dot.view(rows),
        keep.view(rows, hidden),
        1536,
        1.1111111111111112,
    )
    partials = column_partial_sums(
        grad_dropout.view(COLUMN_SUM_SPLITS, rows // COLUMN_SUM_SPLITS, hidden)
    )
    linear_bias_grad = sum_column_partials(partials)
    return (
        grad_weighted.view(batch, seq, hidden),
        weight_grad,
        norm_bias_grad,
        grad_dropout,
        grad_dropout.permute(1, 0),
        linear_bias_grad.view(bias_grad_shape),
    )
