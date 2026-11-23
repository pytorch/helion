"""better-benchmark repro var_mean_3ad14c3d031f (hf/infer/OPTForCausalLM).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_0``:
the decoder layer's residual add of two ``[batch * seq, hidden]`` bf16
streams, returned in bf16 as a ``[batch, seq, hidden]`` view, followed by a
LayerNorm over the hidden dim of the fp32 sum. Each row's population
``var_mean`` is computed in two passes (``mean = sum(x) / hidden``,
``var = sum((x - mean)**2) / hidden``) and
``(x - mean) * rsqrt(var + 1e-5) * weight + bias`` is stored in bf16; the
graph returns that buffer three times, as three ``[batch * seq, hidden]``
views. The kernel writes only the two bf16 outputs; the statistics stay in
registers.
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
        y = centered * torch.rsqrt(var + 1e-5)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return add_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, residual, weight, bias = inputs[:4]
    # _shape_param_0 = [batch, seq, hidden]; _shape_param_1.._3 are the three
    # [batch * seq, hidden] views of the normalized buffer.
    add_shape, out_shape_1, out_shape_2, out_shape_3 = inputs[4:8]
    add_out, norm_out = residual_add_layernorm(x, residual, weight, bias)
    return (
        add_out.view(add_shape),
        norm_out.view(out_shape_1),
        norm_out.view(out_shape_2),
        norm_out.view(out_shape_3),
    )
