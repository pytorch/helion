"""better-benchmark repro mean_f667cd8901af (genai/static/RMSNormForward).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_mean_mul_pow_rsqrt_0``: an RMSNorm
over the last dim of a bf16 ``[M, N]`` input with an fp32 ``[N]`` weight. Each
row is loaded and upcast to fp32, the mean of its squares (``sum(x * x) / N``)
plus eps 1e-6 goes through ``rsqrt``, and ``(x * rsqrt) * weight`` is stored in
bf16. Only the bf16 output is written; the row statistic stays in registers.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        mean_sq = torch.mean(row * row, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + 1e-6)
        y = row * inv_rms * weight[None, :].to(torch.float32)
        out[tile_m, :] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, weight = inputs[:2]
    out = rms_norm_fwd(x, weight)
    return (out,)
