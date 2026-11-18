"""better-benchmark repro sum_139a3020e8bf (timm ViTs, hf LM heads; training).

The exact (erf) GELU backward of an MLP's bf16 activation and the bias
gradient of the linear layer before it: ``grad = grad_out * (0.5 * (erf(x *
0.7071067811865476) + 1) + x * (exp(x * x * -0.5) * 0.3989422804014327))``
(the upstream gradient times the normal CDF plus ``x`` times the normal PDF)
stored in bf16 as ``[rows, hidden]`` and returned with its transposed view,
plus its column sums over the rows in fp32. Inductor emits three kernels:

- ``triton_poi_fused_add_convert_element_type_erf_exp_mul_view_0``: the
  pointwise GELU backward. Inputs and output are dense, so Inductor indexes
  them as one flat dimension (two loads and one store per element); the port
  tiles the same flat views.
- ``triton_red_fused_add_convert_element_type_erf_exp_mul_sum_view_1``: the
  first level of a split column sum. The rows of the bf16 gradient are cut
  into blocks of ``rows_per_split`` (656) rows, the last one partial, and a
  looped reduction writes each block's per-column sum to an fp32
  ``[splits, hidden]`` workspace.
- ``triton_per_fused_add_convert_element_type_erf_exp_mul_sum_view_2``: a
  persistent reduction of the partial sums per column into the fp32
  ``[hidden]`` output. As in Inductor (no ``emulate_precision_casts``), the
  graph's bf16 round trip of that sum stays in fp32.

The split is Inductor's at shape point c064ba70: 50 blocks of 656 rows
(32768 rows). Elsewhere the port keeps 656-row blocks. The blocks are not
whole tensor dims (the last one is partial), so the split kernel loops over
a block's rows with an explicit ``hl.tile`` and masked loads, as Inductor's
looped split reduction does.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split at shape point c064ba70: blocks of 656 rows (50 of them).
ROWS_PER_SPLIT = 656


@helion.kernel(static_shapes=True)
def gelu_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # grad_out, x: [rows * hidden] bf16, flat views of the dense inputs.
    n = x.size(0)
    out = torch.empty([n], dtype=grad_out.dtype, device=grad_out.device)
    for tile_n in hl.tile(n):
        g = grad_out[tile_n].to(torch.float32)
        v = x[tile_n].to(torch.float32)
        cdf = (torch.erf(v * 0.7071067811865476) + 1) * 0.5
        pdf = torch.exp(v * v * -0.5) * 0.3989422804014327
        out[tile_n] = (g * (cdf + v * pdf)).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def column_partial_sums(
    grad: torch.Tensor, rows_per_split: hl.constexpr
) -> torch.Tensor:
    # grad: [rows, hidden] bf16; returns the per-column sums of each block of
    # rows_per_split rows as fp32 [splits, hidden].
    rows, hidden = grad.shape
    splits = (rows + rows_per_split - 1) // rows_per_split
    partials = torch.empty([splits, hidden], dtype=torch.float32, device=grad.device)
    for tile_s, tile_c in hl.tile([splits, hidden]):
        total = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            values = hl.load(grad, [row, tile_c], extra_mask=mask).to(torch.float32)
            values = torch.where(mask, values, 0.0)
            total = total + torch.sum(values, dim=1)
        partials[tile_s, tile_c] = total
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
    grad = gelu_backward(grad_out.view(-1), x.view(-1)).view(rows, hidden)
    partials = column_partial_sums(grad, ROWS_PER_SPLIT)
    bias_grad = sum_partials(partials)
    return grad, grad.permute(1, 0), bias_grad
