"""better-benchmark repro sum_sum_sum_65c3d9f36fd9 (hf/train/MegatronBertForCausalLM).

MegatronBERT training backward: a LayerNorm backward added to another
gradient of the same activations (the graph reads like a block's skip
connection), followed by a dropout backward and the column sums of its
result (the bias gradient of the linear layer before the dropout). The
upstream gradient ``g`` of the LayerNorm output is the sum of three bf16
``[rows, hidden]`` contributions (``rows = batch * seq``), ``(g0 + g1) + g2``
in fp32. With the fp32 LayerNorm weight ``w`` ``[hidden]``, the saved fp32
normalized input ``x_hat`` ``[batch, seq, hidden]``, the fp32 per-row
``scale`` ``[batch, seq, 1]``, the fp32 gradient ``residual``
``[batch, seq, hidden]`` and the bool dropout mask ``keep``:

- ``sum(g * x_hat, (0, 1))`` and ``sum(g, (0, 1))`` in fp32 (the LayerNorm
  weight and bias gradients),
- ``sum_1 = sum(g * w, -1)`` and ``sum_2 = sum(g * w * x_hat, -1)``,
- the summed gradient
  ``add = residual + scale * ((g * w * 1024 - sum_1) - x_hat * sum_2)`` in
  fp32 (``1024`` is the graph literal, the hidden size of the captured
  MegatronBERT),
- the dropout backward ``add * (keep * 1.1111111111111112)`` in bf16 as
  ``[rows, hidden]``, and its transposed view,
- the fp32 column sum of that bf16 tensor over the rows.

Inductor emits four kernels and finishes two of the sums with ATen
reductions:

- ``triton_per_fused_add_convert_element_type_mul_sum_view_0``: a mix-order
  reduction (the ``triton_`` launches in the dump are Inductor's benchmarking
  candidates for the split size): each program owns ``RSPLIT_SIZE = 16``
  consecutive rows and loops over them in row blocks, summing the three
  gradient contributions, computing the two per-row persistent reductions
  over ``hidden`` into two fp32 ``[rows]`` buffers, and accumulating the
  per-column partials of ``g * x_hat`` and ``g`` over its rows, which it
  stores to its rows of the two ``[rows / 16, hidden]`` halves of one fp32
  ``[2, rows / 16, hidden]`` workspace.
- Inductor's wrapper then sums each half of the workspace over its rows with
  an ATen ``workspace.sum(dim=0)``; the port does each of the two column sums
  with a launch of ``sum_gradient_partials`` instead of on the host.
- ``triton_poi_fused_add_convert_element_type_mul_sub_view_1``: pointwise
  ``add`` from ``g`` recomputed from its three contributions, the weight,
  the two per-row sums, ``x_hat``, ``scale`` and ``residual``, stored in
  fp32, plus the bf16 dropout backward.
- ``triton_red_fused_convert_element_type_mul_sum_view_2``: the first level
  of a split column sum of the bf16 dropout backward: a looped reduction
  writes the per-column sum of each of 64 blocks of ``rows / 64`` (128) rows
  to an fp32 ``[64, hidden]`` workspace.
- ``triton_per_fused_convert_element_type_mul_sum_view_3``: a persistent
  reduction of those partials per column into the fp32 ``[hidden]`` output.

As in Inductor (no ``emulate_precision_casts``), the math stays in fp32
between loads and stores: the dropout backward multiplies the fp32 ``add``
by the fp32 scaled mask and rounds once to bf16 (the graph rounds ``add`` and
the scaled mask to bf16 first), and the graph's bf16 round trip of the last
column sum is not materialized.

Both splits divide the rows evenly at the repro's shape point, so the
reductions read ``[splits, rows_per_split, hidden]`` views of the dense
buffers, and the split column sum sums a whole dim of its view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split at shape point e3ecf7e0: 16 rows per
# workspace row (8192 rows -> two [512, 1024] workspace halves).
RSPLIT_SIZE = 16
# Inductor's split of the dropout-backward column sum at that shape point:
# 64 blocks of rows / 64 (128) rows.
COLUMN_SUM_SPLITS = 64


@helion.kernel(static_shapes=True)
def layernorm_bwd_row_sums_partials(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_c: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # grad_a, grad_b, grad_c (bf16) and x_hat (fp32):
    # [splits, rows_per_split, hidden] views of the [rows, hidden] /
    # [batch, seq, hidden] inputs; weight: [hidden]. workspace[0] holds the
    # weight-gradient partials and workspace[1] the bias-gradient partials,
    # as in Inductor's workspace.
    splits, rows_per_split, hidden = grad_a.shape
    row_sum = torch.empty(
        [splits, rows_per_split], dtype=torch.float32, device=grad_a.device
    )
    row_dot = torch.empty(
        [splits, rows_per_split], dtype=torch.float32, device=grad_a.device
    )
    workspace = torch.empty(
        [2, splits, hidden], dtype=torch.float32, device=grad_a.device
    )
    for tile_s in hl.tile(splits):
        weight_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        bias_acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            g = (
                grad_a[tile_s, tile_r, :].to(torch.float32)
                + grad_b[tile_s, tile_r, :].to(torch.float32)
            ) + grad_c[tile_s, tile_r, :].to(torch.float32)
            gw = g * weight[None, None, :]
            xh = x_hat[tile_s, tile_r, :]
            row_sum[tile_s, tile_r] = torch.sum(gw, dim=-1)
            row_dot[tile_s, tile_r] = torch.sum(gw * xh, dim=-1)
            weight_acc = weight_acc + torch.sum(g * xh, dim=1)
            bias_acc = bias_acc + torch.sum(g, dim=1)
        workspace[0, tile_s, :] = weight_acc
        workspace[1, tile_s, :] = bias_acc
    return row_sum, row_dot, workspace


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
def layernorm_bwd_residual_dropout(
    residual: torch.Tensor,
    scale: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_c: torch.Tensor,
    weight: torch.Tensor,
    row_sum: torch.Tensor,
    x_hat: torch.Tensor,
    row_dot: torch.Tensor,
    keep: torch.Tensor,
    norm_size: hl.constexpr,
    keep_scale: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # residual, x_hat (fp32), grad_a, grad_b, grad_c (bf16) and keep (bool):
    # [rows, hidden]; weight: [hidden]; scale, row_sum, row_dot: [rows] fp32.
    # Returns the summed gradient in fp32 and its dropout backward in bf16,
    # both [rows, hidden].
    rows, hidden = grad_a.shape
    grad_residual = torch.empty(
        [rows, hidden], dtype=torch.float32, device=grad_a.device
    )
    grad_dropout = torch.empty([rows, hidden], dtype=grad_a.dtype, device=grad_a.device)
    for tile_r, tile_h in hl.tile([rows, hidden]):
        g = (
            grad_a[tile_r, tile_h].to(torch.float32)
            + grad_b[tile_r, tile_h].to(torch.float32)
        ) + grad_c[tile_r, tile_h].to(torch.float32)
        gw = g * weight[tile_h]
        centered = gw * norm_size - row_sum[tile_r][:, None]
        total = residual[tile_r, tile_h] + scale[tile_r][:, None] * (
            centered - x_hat[tile_r, tile_h] * row_dot[tile_r][:, None]
        )
        kept = keep[tile_r, tile_h].to(torch.float32) * keep_scale
        grad_residual[tile_r, tile_h] = total
        grad_dropout[tile_r, tile_h] = (total * kept).to(grad_dropout.dtype)
    return grad_residual, grad_dropout


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
    grad_a, grad_b, grad_c, weight, x_hat, scale, residual, keep = inputs[:8]
    batch, seq, hidden = inputs[8]  # _shape_param_0 = [batch, seq, hidden]
    rows, _ = inputs[11]  # _shape_param_3 = [batch * seq, hidden]
    bias_grad_shape = inputs[12]  # _shape_param_4 = [hidden]
    splits = rows // RSPLIT_SIZE
    row_sum, row_dot, workspace = layernorm_bwd_row_sums_partials(
        grad_a.view(splits, RSPLIT_SIZE, hidden),
        grad_b.view(splits, RSPLIT_SIZE, hidden),
        grad_c.view(splits, RSPLIT_SIZE, hidden),
        weight,
        x_hat.view(splits, RSPLIT_SIZE, hidden),
    )
    weight_grad = sum_gradient_partials(workspace[0])
    norm_bias_grad = sum_gradient_partials(workspace[1])
    grad_residual, grad_dropout = layernorm_bwd_residual_dropout(
        residual.view(rows, hidden),
        scale.view(rows),
        grad_a,
        grad_b,
        grad_c,
        weight,
        row_sum.view(rows),
        x_hat.view(rows, hidden),
        row_dot.view(rows),
        keep.view(rows, hidden),
        1024,
        1.1111111111111112,
    )
    partials = column_partial_sums(
        grad_dropout.view(COLUMN_SUM_SPLITS, rows // COLUMN_SUM_SPLITS, hidden)
    )
    linear_bias_grad = sum_column_partials(partials)
    return (
        weight_grad,
        norm_bias_grad,
        grad_residual.view(batch, seq, hidden),
        grad_dropout,
        grad_dropout.permute(1, 0),
        linear_bias_grad.view(bias_grad_shape),
    )
