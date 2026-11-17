"""better-benchmark repro var_mean_f7906a6c6957 (genai/static/LayerNormForward).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_0``: a
bias-free LayerNorm forward over the last dim of a bf16 ``[rows, hidden]``
input. Each row is promoted to fp32, its population ``var_mean`` is computed in
two passes (``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``),
and ``(x - mean) * rsqrt(var + 1e-6) * weight`` (fp32 weight) is stored in bf16.
Only the normalized output is written; the statistics stay in registers.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def layernorm_no_bias_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(row, dim=-1, keepdim=True)
        centered = row - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-6)
        y = y * weight[None, :]
        out[tile_r, :] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, weight = inputs[:2]
    out = layernorm_no_bias_fwd(x, weight)
    return (out,)
