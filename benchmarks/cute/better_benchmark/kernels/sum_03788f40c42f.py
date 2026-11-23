"""better-benchmark repro sum_03788f40c42f (hf GPT2ForSequenceClassification / DistillGPT2).

The tanh-approximate GELU backward of an MLP's bf16 activation and the bias
gradient of the linear layer before it. With ``t = tanh((x + x * x * x *
0.044715) * 0.7978845608028654)`` and ``d = grad_out * (x * 0.5) * (1 - t * t)
* 0.7978845608028654``, the gradient is ``(d + d * 0.044715 * (x * x * 3.0)) +
grad_out * (t + 1) * 0.5``, stored in bf16 as ``[rows, hidden]``, plus its
column sums over the rows in fp32. This is sum_851ae8199532's graph (GPT-Neo)
without the transposed view of the gradient. Inductor emits three kernels,
which the port reproduces:

- ``triton_poi_fused_add_convert_element_type_mul_pow_sub_tanh_view_0``: the
  pointwise GELU backward. Inputs and output are dense, so Inductor indexes
  them as one flat dimension (two loads and one store per element); the port
  tiles the same flat views. As in Inductor (no ``emulate_precision_casts``),
  the graph's bf16 round trips of intermediates (``x * 0.5``, the three terms
  and their first sum) stay in fp32, and the powers are multiplies (``x * x``
  shared by the cube and the square).
- ``triton_red_fused_sum_view_1``: the first level of a split column sum. The
  rows of the bf16 gradient are cut into ``SPLITS`` (64) blocks, and a looped
  reduction writes each block's per-column sum to an fp32 ``[splits, hidden]``
  workspace.
- ``triton_per_fused_convert_element_type_sum_view_2``: a persistent
  reduction of the partial sums per column into the fp32 ``[hidden]`` output.
  The graph's bf16 round trip of that sum stays in fp32 too.

The split is Inductor's: 64 blocks at both shape points, of 128 rows at
58ff2bc5 (8192 rows, the point this port was checked at) and of 256 rows at
05a5e903 (16384 rows). The blocks divide the rows evenly, so the split kernel
reads a ``[splits, rows_per_split, hidden]`` view of the gradient and sums a
whole dim of it; ``forward()`` rejects row counts the blocks do not divide.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split: 64 blocks of rows (128 rows each at shape point 58ff2bc5,
# 256 at 05a5e903).
SPLITS = 64


@helion.kernel(static_shapes=True)
def gelu_tanh_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # grad_out, x: [rows * hidden] bf16, flat views of the dense inputs.
    n = x.size(0)
    out = torch.empty([n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_n in hl.tile(n):
        g = grad_out[tile_n].to(torch.float32)
        v = x[tile_n].to(torch.float32)
        g_half_v = g * (v * 0.5)
        square = v * v
        cube = square * v
        t = torch.tanh((v + cube * 0.044715) * 0.7978845608028654)
        d = g_half_v * (1 - t * t) * 0.7978845608028654
        cubic = (d * 0.044715) * (square * 3.0)
        out[tile_n] = ((d + cubic) + g * (t + 1) * 0.5).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def column_partial_sums(grad_rows: torch.Tensor) -> torch.Tensor:
    # grad_rows: [splits, rows_per_split, hidden] bf16, a view of the
    # [rows, hidden] gradient; returns each block's per-column sum as fp32
    # [splits, hidden].
    splits, rows_per_split, hidden = grad_rows.shape
    partials = torch.empty(
        [splits, hidden], dtype=torch.float32, device=grad_rows.device
    )
    for tile_s, tile_c in hl.tile([splits, hidden]):
        partials[tile_s, tile_c] = torch.sum(
            grad_rows[tile_s, :, tile_c].to(torch.float32), dim=1
        )
    return partials


@helion.kernel(static_shapes=True)
def sum_partials(partials: torch.Tensor) -> torch.Tensor:
    # partials: [splits, hidden] fp32; returns their per-column sum as fp32
    # [hidden].
    splits, hidden = partials.shape
    out = torch.empty([hidden], dtype=torch.float32, device=partials.device)
    for tile_c in hl.tile(hidden):
        out[tile_c] = torch.sum(partials[:, tile_c], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_out, x = inputs[:2]
    rows, hidden = inputs[4]  # _shape_param_2 = [B*S, hidden]
    bias_grad_shape = inputs[5]  # _shape_param_3 = [hidden]
    if rows % SPLITS != 0:
        raise ValueError(f"{rows} rows are not {SPLITS} whole blocks")
    grad = gelu_tanh_backward(grad_out.view(-1), x.view(-1)).view(rows, hidden)
    partials = column_partial_sums(grad.view(SPLITS, rows // SPLITS, hidden))
    bias_grad = sum_partials(partials)
    return grad, bias_grad.view(bias_grad_shape)
