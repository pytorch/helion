"""better-benchmark repro sum_2ba0992ca00b (hf GoogleFnet training).

The tanh-approximate GELU backward of an MLP's fp32 activation and the bias
gradient of the linear layer before it. With ``t = tanh((x + x * x * x *
0.044715) * 0.7978845608028654)`` and ``d = grad_out * (x * 0.5) * (1 - t * t)
* 0.7978845608028654``, the gradient is ``(d + d * 0.044715 * (x * x * 3.0)) +
grad_out * (t + 1) * 0.5``, stored in fp32 as ``[rows, hidden]`` and returned
with its transposed view, plus its column sums over the rows. This is
sum_851ae8199532's graph (GPT-Neo, bf16) in fp32, so no kernel casts.
Inductor emits three kernels, which the port reproduces:

- ``triton_poi_fused_add_mul_pow_sub_tanh_view_0``: the pointwise GELU
  backward. Inputs and output are dense, so Inductor indexes them as one flat
  dimension (two loads and one store per element); the port tiles the same
  flat views. The powers are multiplies (``x * x`` shared by the cube and the
  square).
- ``triton_red_fused_add_mul_pow_sub_sum_tanh_view_1``: the first level of a
  split column sum. The rows of the gradient are cut into blocks of
  ``ROWS_PER_SPLIT`` (256) rows, and a looped reduction writes each block's
  per-column sum to a ``[splits, hidden]`` workspace.
- ``triton_per_fused_add_mul_pow_sub_sum_tanh_view_2``: a persistent
  reduction of the partial sums per column into the ``[hidden]`` output.

The split is Inductor's at shape point 94cc1ac5: 64 blocks of 256 rows (16384
rows). The blocks divide the rows evenly there, so the split kernel reads a
``[splits, rows_per_split, hidden]`` view of the gradient and sums a whole dim
of it; ``forward()`` rejects row counts the blocks do not divide.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split at shape point 94cc1ac5: blocks of 256 rows (64 of them).
ROWS_PER_SPLIT = 256


@helion.kernel(static_shapes=True)
def gelu_tanh_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # grad_out, x: [rows * hidden] fp32, flat views of the dense inputs.
    n = x.size(0)
    out = torch.empty([n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_n in hl.tile(n):
        g = grad_out[tile_n]
        v = x[tile_n]
        g_half_v = g * (v * 0.5)
        square = v * v
        cube = square * v
        t = torch.tanh((v + cube * 0.044715) * 0.7978845608028654)
        d = g_half_v * (1 - t * t) * 0.7978845608028654
        cubic = (d * 0.044715) * (square * 3.0)
        out[tile_n] = (d + cubic) + g * (t + 1) * 0.5
    return out


@helion.kernel(static_shapes=True)
def column_partial_sums(grad_rows: torch.Tensor) -> torch.Tensor:
    # grad_rows: [splits, rows_per_split, hidden] fp32, a view of the
    # [rows, hidden] gradient; returns each block's per-column sum as
    # [splits, hidden].
    splits, rows_per_split, hidden = grad_rows.shape
    partials = torch.empty(
        [splits, hidden], dtype=grad_rows.dtype, device=grad_rows.device
    )
    for tile_s, tile_c in hl.tile([splits, hidden]):
        partials[tile_s, tile_c] = torch.sum(grad_rows[tile_s, :, tile_c], dim=1)
    return partials


@helion.kernel(static_shapes=True)
def sum_partials(partials: torch.Tensor) -> torch.Tensor:
    # partials: [splits, hidden] fp32; returns their per-column sum as
    # [hidden].
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=partials.dtype, device=partials.device)
    for tile_c in hl.tile(hidden):
        out[tile_c] = torch.sum(partials[:, tile_c], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_out, x = inputs[:2]
    rows, hidden = inputs[4]  # _shape_param_2 = [B*S, hidden]
    bias_grad_shape = inputs[5]  # _shape_param_3 = [hidden]
    if rows % ROWS_PER_SPLIT != 0:
        raise ValueError(f"{rows} rows are not whole blocks of {ROWS_PER_SPLIT}")
    grad = gelu_tanh_backward(grad_out.view(-1), x.view(-1)).view(rows, hidden)
    partials = column_partial_sums(
        grad.view(rows // ROWS_PER_SPLIT, ROWS_PER_SPLIT, hidden)
    )
    bias_grad = sum_partials(partials)
    return grad, grad.permute(1, 0), bias_grad.view(bias_grad_shape)
