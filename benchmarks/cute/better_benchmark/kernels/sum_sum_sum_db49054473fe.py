"""better-benchmark repro sum_sum_sum_db49054473fe (timm/train/vit_base_patch14_dinov2.lvd142m).

A LayerNorm backward with a residual add, followed by a LayerScale backward,
in a DINOv2 ViT block. With the bf16 upstream gradient ``g`` ``[rows, H]``
(``rows = B*T``), the fp32 LayerNorm weight ``w`` ``[H]``, the fp32 normalized
activations ``x_hat`` and residual gradient ``res`` ``[B, T, H]``, the fp32
per-row ``scale`` ``[B, T, 1]`` (strided), the bf16 LayerScale input ``x``
``[rows, H]`` and the fp32 LayerScale weight ``gamma`` ``[H]``:

- ``sum(g * x_hat, (0, 1))`` and ``sum(g, (0, 1))`` in fp32 (the LayerNorm
  weight and bias gradients),
- ``add = res + scale * ((g * w * 768 - sum(g * w, -1)) - x_hat *
  sum(g * w * x_hat, -1))`` in fp32 (``768`` is the graph literal),
- ``sum(add * x, (0, 1))`` in fp32 (the LayerScale weight gradient),
- ``add * gamma`` in bf16 as ``[rows, H]``, and its transposed view,
- the column sum of that bf16 tensor in fp32.

Inductor emits five kernels and finishes two outputs with ATen reductions:

- ``triton_per_fused_add_convert_element_type_mul_sub_sum_view_0``: a
  mix-order reduction (the ``triton_`` launches in the dump are Inductor's
  benchmarking candidates for the split size): each program owns
  ``RSPLIT_SIZE = 16`` consecutive rows and loops over them in row blocks,
  computing the two per-row persistent reductions over ``H``, storing ``add``
  and ``add * gamma``, and accumulating the per-column partials of
  ``g * x_hat`` and ``g`` over its rows, which it stores to its rows of an
  fp32 ``[2, rows / 16, H]`` workspace (two ``[rows / 16, H]`` tensors in
  the port). ``scale`` is gathered at ``(row // T, row % T)``, since the
  16-row splits do not line up with ``T``.
- Inductor's wrapper then sums each half of the workspace over its rows with
  an ATen ``workspace.sum(dim=0)``; the port does these two column sums with
  two launches of a kernel instead of on the host.
- ``triton_red_fused_mul_sum_view_1``: the first level of the split reduction
  of ``add * x`` over the rows: blocks of ``ROWS_PER_SPLIT = 886`` rows (the
  last one partial) write their per-column partial to an fp32
  ``[splits, H]`` workspace.
- ``triton_red_fused_mul_sum_view_2``: sums those partials per column.
- ``triton_red_fused_convert_element_type_mul_sum_view_3``: the same split
  reduction over the bf16 ``add * gamma``. Inductor writes these partials
  into the first split's workspace memory, so ``forward()`` allocates that
  workspace once and both first-level kernels store into it.
- ``triton_red_fused_convert_element_type_mul_sum_view_4``: sums those
  partials per column. Like Inductor, the graph's bf16 round trip of this
  sum stays in fp32.

The split blocks of the two looped reductions are not whole tensor dims (the
last block is partial), so those kernels loop over a block's rows with an
explicit ``hl.tile`` and masked loads, as Inductor's looped reductions do.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's mix-order reduction split for this shape: 16 rows per workspace
# row (175360 rows -> two [10960, 768] workspace halves).
RSPLIT_SIZE = 16
# Inductor's split of the two looped reductions over the rows for this shape:
# 198 blocks of 886 rows, the last one partial.
ROWS_PER_SPLIT = 886


@helion.kernel(static_shapes=True)
def layernorm_bwd_residual_layerscale_partials(
    grad: torch.Tensor,
    weight: torch.Tensor,
    x_hat: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    norm_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # grad (bf16), x_hat, residual (fp32): [splits, rows_per_split, hidden]
    # views of the [rows, hidden] / [batch, seq, hidden] inputs; weight,
    # gamma: [hidden]; scale: [batch, seq] (strided view of [batch, seq, 1]).
    splits, rows_per_split, hidden = grad.shape
    seq = scale.size(1)
    grad_sum = torch.empty(
        [splits, rows_per_split, hidden], dtype=torch.float32, device=grad.device
    )
    scaled = torch.empty(
        [splits, rows_per_split, hidden], dtype=torch.bfloat16, device=grad.device
    )
    dot_partial = torch.empty([splits, hidden], dtype=torch.float32, device=grad.device)
    grad_partial = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad.device
    )
    for tile_s in hl.tile(splits):
        dot = hl.zeros([tile_s, hidden], dtype=torch.float32)
        total = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            g = grad[tile_s, tile_r, :].to(torch.float32)
            gw = g * weight[None, None, :]
            xh = x_hat[tile_s, tile_r, :]
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            r = scale[row // seq, row % seq][:, :, None]
            row_sum = torch.sum(gw, dim=-1, keepdim=True)
            row_dot = torch.sum(gw * xh, dim=-1, keepdim=True)
            y = residual[tile_s, tile_r, :] + r * (
                (gw * norm_size - row_sum) - xh * row_dot
            )
            grad_sum[tile_s, tile_r, :] = y
            scaled[tile_s, tile_r, :] = (y * gamma[None, None, :]).to(scaled.dtype)
            dot = dot + torch.sum(g * xh, dim=1)
            total = total + torch.sum(g, dim=1)
        dot_partial[tile_s, :] = dot
        grad_partial[tile_s, :] = total
    return grad_sum, scaled, dot_partial, grad_partial


@helion.kernel(static_shapes=True)
def workspace_column_sum(partial: torch.Tensor) -> torch.Tensor:
    # Inductor's ATen ``workspace.sum(dim=0)`` over one workspace half.
    splits, hidden = partial.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partial.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partial[:, tile_h], dim=0)
    return out


@helion.kernel(static_shapes=True)
def layerscale_weight_grad_partials(
    grad: torch.Tensor,
    x: torch.Tensor,
    partial: torch.Tensor,
    rows_per_split: hl.constexpr,
) -> torch.Tensor:
    # grad: [rows, hidden] fp32; x: [rows, hidden] bf16; partial:
    # [splits, hidden] fp32, the workspace.
    rows, hidden = grad.shape
    splits = partial.size(0)
    for tile_s, tile_h in hl.tile([splits, hidden]):
        acc = hl.zeros([tile_s, tile_h], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            g = hl.load(grad, [row, tile_h], extra_mask=mask)
            xf = hl.load(x, [row, tile_h], extra_mask=mask).to(torch.float32)
            acc = acc + torch.sum(g * xf, dim=1)
        partial[tile_s, tile_h] = acc
    return partial


@helion.kernel(static_shapes=True)
def layerscale_weight_grad(partial: torch.Tensor) -> torch.Tensor:
    splits, hidden = partial.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partial.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partial[:, tile_h], dim=0)
    return out


@helion.kernel(static_shapes=True)
def scaled_grad_column_partials(
    scaled: torch.Tensor, partial: torch.Tensor, rows_per_split: hl.constexpr
) -> torch.Tensor:
    # scaled: [rows, hidden] bf16; partial: [splits, hidden] fp32, the
    # workspace.
    rows, hidden = scaled.shape
    splits = partial.size(0)
    for tile_s, tile_h in hl.tile([splits, hidden]):
        acc = hl.zeros([tile_s, tile_h], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            values = hl.load(scaled, [row, tile_h], extra_mask=valid[:, :, None])
            acc = acc + torch.sum(values.to(torch.float32), dim=1)
        partial[tile_s, tile_h] = acc
    return partial


@helion.kernel(static_shapes=True)
def scaled_grad_column_sum(partial: torch.Tensor) -> torch.Tensor:
    # The graph rounds this sum to bf16 and back; Inductor keeps it in fp32.
    splits, hidden = partial.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partial.device)
    for tile_h in hl.tile(hidden):
        out[tile_h] = torch.sum(partial[:, tile_h], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, weight, x_hat, scale, residual, x, gamma = inputs[:7]
    batch, seq, hidden = inputs[7]  # _shape_param_0 = [batch, seq, hidden]
    rows = batch * seq
    splits = rows // RSPLIT_SIZE
    grad_sum, scaled, dot_partial, grad_partial = (
        layernorm_bwd_residual_layerscale_partials(
            grad.view(splits, RSPLIT_SIZE, hidden),
            weight,
            x_hat.view(splits, RSPLIT_SIZE, hidden),
            scale.view(batch, seq),
            residual.view(splits, RSPLIT_SIZE, hidden),
            gamma,
            768,
        )
    )
    weight_grad = workspace_column_sum(dot_partial)
    bias_grad = workspace_column_sum(grad_partial)
    grad_sum = grad_sum.view(rows, hidden)
    scaled = scaled.view(rows, hidden)
    blocks = (rows + ROWS_PER_SPLIT - 1) // ROWS_PER_SPLIT
    # One workspace for both split reductions, as in Inductor.
    partial = torch.empty([blocks, hidden], dtype=torch.float32, device=grad.device)
    gamma_grad = layerscale_weight_grad(
        layerscale_weight_grad_partials(grad_sum, x, partial, ROWS_PER_SPLIT)
    )
    scaled_sum = scaled_grad_column_sum(
        scaled_grad_column_partials(scaled, partial, ROWS_PER_SPLIT)
    )
    return (
        weight_grad,
        bias_grad,
        grad_sum.view(batch, seq, hidden),
        gamma_grad,
        scaled,
        scaled.permute(1, 0),
        scaled_sum,
    )
