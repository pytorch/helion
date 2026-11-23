"""better-benchmark repro sum_sum_sum_8a3378108574 (hf GPT-Neo causal-LM / sequence-classification; training).

A LayerNorm backward with a residual add, feeding the backward of the linear
layer before it. The upstream gradient ``g`` of the LayerNorm output is the
sum of three bf16 ``[rows, H]`` contributions (``rows = B*T``),
``(g0 + g1) + g2`` in fp32. With the fp32 LayerNorm weight ``w`` ``[H]``, the
fp32 normalized activations ``x_hat`` and residual gradient ``res``
``[B, T, H]`` and the fp32 per-row ``scale`` ``[B, T, 1]``:

- ``sum(g * x_hat, (0, 1))`` and ``sum(g, (0, 1))`` in fp32 (the LayerNorm
  weight and bias gradients),
- ``add = res + scale * ((g * w * 2048 - sum(g * w, -1)) - x_hat *
  sum(g * w * x_hat, -1))`` in fp32 (``2048`` is the graph literal, the hidden
  size of the captured GPT-Neo),
- ``add`` in bf16 as ``[rows, H]``, and its transposed view,
- the column sum of that bf16 tensor in fp32.

Inductor emits three kernels and finishes two outputs with ATen reductions:

- ``triton_per_fused_add_convert_element_type_mul_sub_sum_view_0``: a
  mix-order reduction (the ``triton_`` launches in the dump are Inductor's
  benchmarking candidates for the split size): each program owns
  ``RSPLIT_SIZE = 8`` consecutive rows and loops over them in row blocks,
  summing the three gradient contributions, computing the two per-row
  persistent reductions over ``H``, storing ``add`` in fp32 and in bf16, and
  accumulating the per-column partials of ``g * x_hat`` and ``g`` over its
  rows, which it stores to its rows of the two halves of one fp32
  ``[2, rows / 8, H]`` workspace.
- Inductor's wrapper then sums each half of the workspace over its rows with
  an ATen ``workspace.sum(dim=0)``; the port does these two column sums with
  two launches of a kernel, on views of the halves, instead of on the host.
- ``triton_red_fused_convert_element_type_sum_view_1``: the first level of the
  split reduction of the bf16 ``add`` over the rows: blocks of
  ``ROWS_PER_SPLIT = 128`` consecutive rows write their per-column partial to
  an fp32 ``[rows / 128, H]`` workspace (a looped reduction).
- ``triton_per_fused_convert_element_type_sum_view_2``: sums those partials
  per column (a persistent reduction). Like Inductor, the graph's bf16 round
  trip of this sum stays in fp32.

Both splits divide the rows evenly at the repro's shape point, so the kernels
read ``[splits, rows_per_split, H]`` views of the dense inputs, and the split
reduction sums a whole dim of its view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split for shape point b9b3dbcc: 8 rows per
# workspace row (4096 rows -> a [2, 512, 2048] workspace).
RSPLIT_SIZE = 8
# Inductor's split of the looped reduction over the rows for this shape:
# 32 blocks of 128 rows.
ROWS_PER_SPLIT = 128


@helion.kernel(static_shapes=True)
def layernorm_bwd_residual_partials(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    grad_c: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # grad_a, grad_b, grad_c (bf16), x_hat, residual (fp32):
    # [splits, rows_per_split, hidden] views of the [rows, hidden] /
    # [batch, seq, hidden] inputs; weight: [hidden]; scale:
    # [splits, rows_per_split], a view of [batch, seq, 1].
    # workspace: Inductor's [2, splits, hidden] buffer; half 0 holds the
    # partials of g * x_hat, half 1 those of g.
    splits, rows_per_split, hidden = grad_a.shape
    grad_sum = torch.empty(
        [splits, rows_per_split, hidden], dtype=torch.float32, device=grad_a.device
    )
    grad_sum_lowp = torch.empty(
        [splits, rows_per_split, hidden], dtype=grad_a.dtype, device=grad_a.device
    )
    workspace = torch.empty(
        [2, splits, hidden], dtype=torch.float32, device=grad_a.device
    )
    for tile_s in hl.tile(splits):
        dot = hl.zeros([tile_s, hidden], dtype=torch.float32)
        total = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            g = (
                grad_a[tile_s, tile_r, :].to(torch.float32)
                + grad_b[tile_s, tile_r, :].to(torch.float32)
            ) + grad_c[tile_s, tile_r, :].to(torch.float32)
            gw = g * weight[None, None, :]
            xh = x_hat[tile_s, tile_r, :]
            r = scale[tile_s, tile_r][:, :, None]
            row_sum = torch.sum(gw, dim=-1, keepdim=True)
            row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
            y = residual[tile_s, tile_r, :] + r * (
                (gw * norm_size - row_sum) - xh * row_dot
            )
            grad_sum[tile_s, tile_r, :] = y
            grad_sum_lowp[tile_s, tile_r, :] = y.to(grad_sum_lowp.dtype)
            dot = dot + torch.sum(g * xh, dim=1)
            total = total + torch.sum(g, dim=1)
        workspace[0, tile_s, :] = dot
        workspace[1, tile_s, :] = total
    return grad_sum, grad_sum_lowp, workspace


@helion.kernel(static_shapes=True)
def workspace_column_sum(partial: torch.Tensor) -> torch.Tensor:
    # Inductor's ATen ``workspace.sum(dim=0)`` over one workspace half;
    # partial: [splits, hidden], a view of that half.
    splits, hidden = partial.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partial.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partial[:, tile_h], dim=0)
    return out


@helion.kernel(static_shapes=True)
def grad_column_partials(grad_blocks: torch.Tensor) -> torch.Tensor:
    # grad_blocks: [splits, rows_per_split, hidden] bf16, a view of the
    # [rows, hidden] gradient.
    splits, rows_per_split, hidden = grad_blocks.shape
    partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_blocks.device
    )
    for tile_s, tile_h in hl.tile([splits, hidden]):
        partial[tile_s, tile_h] = torch.sum(
            grad_blocks[tile_s, :, tile_h].to(torch.float32), dim=1
        )
    return partial


@helion.kernel(static_shapes=True)
def grad_column_sum(partial: torch.Tensor) -> torch.Tensor:
    # The graph rounds this sum to bf16 and back; Inductor keeps it in fp32.
    splits, hidden = partial.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partial.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partial[:, tile_h], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_a, grad_b, grad_c, weight, x_hat, scale, residual = inputs[:7]
    batch, seq, hidden = inputs[7]  # _shape_param_0 = [batch, seq, hidden]
    rows, _ = inputs[10]  # _shape_param_3 = [batch * seq, hidden]
    column_sum_shape = inputs[11]  # _shape_param_4 = [hidden]
    splits = rows // RSPLIT_SIZE
    grad_sum, grad_sum_lowp, workspace = layernorm_bwd_residual_partials(
        grad_a.view(splits, RSPLIT_SIZE, hidden),
        grad_b.view(splits, RSPLIT_SIZE, hidden),
        grad_c.view(splits, RSPLIT_SIZE, hidden),
        weight,
        x_hat.view(splits, RSPLIT_SIZE, hidden),
        scale.view(splits, RSPLIT_SIZE),
        residual.view(splits, RSPLIT_SIZE, hidden),
        2048,
    )
    weight_grad = workspace_column_sum(workspace[0])
    bias_grad = workspace_column_sum(workspace[1])
    grad_sum_lowp = grad_sum_lowp.view(rows, hidden)
    column_sum = grad_column_sum(
        grad_column_partials(
            grad_sum_lowp.view(rows // ROWS_PER_SPLIT, ROWS_PER_SPLIT, hidden)
        )
    )
    return (
        weight_grad,
        bias_grad,
        grad_sum.view(batch, seq, hidden),
        grad_sum_lowp,
        grad_sum_lowp.permute(1, 0),
        column_sum.view(column_sum_shape),
    )
