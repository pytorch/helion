"""better-benchmark repro var_mean_sum_53c829d32b0c (genai/static/LayerNormBackward).

A bias-free LayerNorm over the last dim of a bf16 ``[rows, hidden]`` input
with an fp32 ``[hidden]`` weight, returning the fp32 row mean and
``rsqrt(var + 1e-6)`` (both ``[rows, 1]``) and the bf16 scalar sum of the
normalized, weighted output. Inductor emits two kernels:

- ``triton_per_fused_add_convert_element_type_mul_rsqrt_sub_sum_var_mean_0``:
  a persistent reduction per row. The row is promoted to fp32, its population
  ``var_mean`` is computed in two passes (``mean = sum(x) / hidden``,
  ``var = sum((x - mean)**2) / hidden``), the mean and ``rsqrt(var + 1e-6)``
  are stored, and the row sum of ``(x - mean) * rsqrt * weight`` goes to an
  fp32 ``[rows]`` workspace. Like Inductor (no ``emulate_precision_casts``),
  the graph's bf16 rounding of that product before the sum stays in fp32.
- ``triton_red_fused_convert_element_type_mul_sub_sum_1``: a looped reduction
  (one program) summing the per-row workspace into the bf16 scalar.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def layernorm_stats_row_sums(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [rows, hidden] bf16; weight: [hidden] fp32.
    rows = x.size(0)
    mean_out = torch.empty([rows], dtype=torch.float32, device=x.device)
    rstd_out = torch.empty([rows], dtype=torch.float32, device=x.device)
    row_sums = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(row, dim=-1)
        centered = row - mean[:, None]
        var = torch.mean(centered * centered, dim=-1)
        rstd = torch.rsqrt(var + 1e-6)
        mean_out[tile_r] = mean
        rstd_out[tile_r] = rstd
        y = centered * rstd[:, None] * weight[None, :]
        row_sums[tile_r] = torch.sum(y, dim=-1)
    return mean_out, rstd_out, row_sums


@helion.kernel(static_shapes=True)
def sum_row_sums(row_sums: torch.Tensor) -> torch.Tensor:
    # row_sums: [1, rows] fp32, a view of the per-row workspace.
    one, rows = row_sums.shape
    out = torch.empty([one], dtype=torch.bfloat16, device=row_sums.device)
    for tile in hl.tile(one):
        out[tile] = torch.sum(row_sums[tile, :], dim=-1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, weight = inputs[:2]
    rows = x.size(0)
    mean, rstd, row_sums = layernorm_stats_row_sums(x, weight)
    total = sum_row_sums(row_sums.view(1, rows))
    return mean.view(rows, 1), rstd.view(rows, 1), total.view(())
