"""better-benchmark repro var_mean_f4578c128d74 (timm/infer/deit_tiny_patch16_224.fb_in1k).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_0``:
the transformer block's residual add (the ``[batch, seq, hidden]`` stream plus
a ``[batch * seq, hidden]`` projection output viewed to the same shape),
returned in bf16, followed by a LayerNorm over the hidden dim of the fp32 sum.
Each row's population ``var_mean`` is computed in two passes
(``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``) and
``(x - mean) * rsqrt(var + 1e-6) * weight + bias`` is stored in bf16 as a
``[batch * seq, hidden]`` view. The kernel writes only the two bf16 outputs;
the statistics stay in registers.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    add_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        summed = x[tile_r, :].to(torch.float32) + residual[tile_r, :].to(torch.float32)
        add_out[tile_r, :] = summed.to(add_out.dtype)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-6)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return add_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    residual, x, weight, bias = inputs[:4]
    out_shape = inputs[5]  # _shape_param_1 = [batch * seq, hidden] (or [-1, hidden])
    batch, seq, hidden = x.shape
    add_out, norm_out = residual_add_layernorm(
        x.view(batch * seq, hidden), residual.view(batch * seq, hidden), weight, bias
    )
    return add_out.view(batch, seq, hidden), norm_out.view(out_shape)
