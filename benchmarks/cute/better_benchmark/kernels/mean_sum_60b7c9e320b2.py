"""better-benchmark repro mean_sum_60b7c9e320b2 (genai RMSNormBackward, static).

An RMSNorm over the last dim of a bf16 ``[M, N]`` input with an fp32 ``[N]``
weight that returns the fp32 ``[M, 1]`` ``rsqrt`` statistic and the bf16 sum of
the whole normalized output. Inductor emits two kernels:

- ``triton_per_fused_add_convert_element_type_mean_mul_pow_rsqrt_sum_0``: a
  persistent reduction per row; loads the row, upcasts it to fp32, takes the
  mean of its squares (``sum(x * x) / N``), adds eps 1e-6 and stores the
  ``rsqrt`` output, then sums ``(x * rsqrt) * weight`` over the row into an
  fp32 ``[M]`` workspace. Like Inductor (no ``emulate_precision_casts``), the
  graph's bf16 cast of the normalized output before the sum stays in fp32.
- ``triton_red_fused_convert_element_type_mul_sum_1``: a looped reduction of
  the ``M`` per-row partials (a ``[1, M]`` view of the workspace reduced over
  its single row) into the 0-d bf16 ``sum_1``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def rms_norm_rsqrt_row_sums(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [M, N] bf16; weight: [N] fp32.
    m, n = x.shape
    inv_rms_out = torch.empty([m, 1], dtype=torch.float32, device=x.device)
    row_sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        mean_sq = torch.mean(row * row, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + 1e-6)
        inv_rms_out[tile_m, :] = inv_rms
        y = row * inv_rms * weight[None, :]
        row_sums[tile_m] = torch.sum(y, dim=-1)
    return inv_rms_out, row_sums


@helion.kernel(static_shapes=True)
def total_sum(row_sums: torch.Tensor) -> torch.Tensor:
    # row_sums: [1, M] view of the per-row partial sums.
    rows, _ = row_sums.shape
    out = torch.empty([rows], dtype=torch.bfloat16, device=row_sums.device)
    for tile_r in hl.tile(rows):
        out[tile_r] = torch.sum(row_sums[tile_r, :], dim=-1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, weight = inputs[:2]
    m = x.shape[0]
    inv_rms, row_sums = rms_norm_rsqrt_row_sums(x, weight)
    total = total_sum(row_sums.view(1, m))
    return inv_rms, total.view([])
