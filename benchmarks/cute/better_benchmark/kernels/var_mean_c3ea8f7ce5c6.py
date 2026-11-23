"""better-benchmark repro var_mean_c3ea8f7ce5c6 (hf causal-LM decoders).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_0``:
a decoder block's residual add (the ``[batch, seq, hidden]`` stream plus a
``[batch * seq, hidden]`` dense output viewed to the same shape) feeding a
LayerNorm over the hidden dim. The sum stays in fp32 (Inductor does not round
it to bf16 before the statistics) and is not stored. Each row's population
``var_mean`` is computed in two passes (``mean = sum(x) / hidden``,
``var = sum((x - mean)**2) / hidden``) and
``(x - mean) * rsqrt(var + 1e-5) * weight + bias`` is stored in bf16. The
kernel writes that one buffer, returned as ``[batch, seq, hidden]`` and as
three ``[batch * seq, hidden]`` views of it; the statistics stay in registers.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # residual, x: [rows, hidden] bf16; weight, bias: [hidden] bf16.
    rows, hidden = x.shape
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        summed = residual[tile_r, :].to(torch.float32) + x[tile_r, :].to(torch.float32)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-5)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, residual, weight, bias = inputs[:4]
    # _shape_param_0 = [batch, seq, hidden]; the other three are
    # [batch * seq, hidden].
    norm_shape, flat_shape_1, flat_shape_2, flat_shape_3 = inputs[4:8]
    batch, seq, hidden = residual.shape
    norm_out = residual_add_layernorm(
        residual.view(batch * seq, hidden), x, weight, bias
    )
    return (
        norm_out.view(norm_shape),
        norm_out.view(flat_shape_1),
        norm_out.view(flat_shape_2),
        norm_out.view(flat_shape_3),
    )
