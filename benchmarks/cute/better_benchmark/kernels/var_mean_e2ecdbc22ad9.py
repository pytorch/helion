"""better-benchmark repro var_mean_e2ecdbc22ad9 (timm ViT/DeiT and hf GPT-Neo training).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_mul_rsqrt_sub_var_mean_view_0``:
a transformer block's residual add (a bf16 ``[batch * seq, hidden]``
projection output viewed to ``[batch, seq, hidden]`` plus the fp32 stream of
that shape) feeding a LayerNorm over the hidden dim, in the training graph
that also returns what the backward needs. The sum is computed and stored in
fp32. Each row's population ``var_mean`` is computed in two passes
(``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``, both true
divisions by the count). The kernel stores four buffers: the fp32 sum, the
fp32 normalized value ``(x - mean) * rsqrt(var + 1e-6)``, that value times
the fp32 weight plus the fp32 bias in bf16 (returned as a
``[batch * seq, hidden]`` view), and the per-row ``rsqrt(var + 1e-6) / 192``
in fp32 ``[batch, seq, 1]``. The divisor 192 is a literal of the graph
(captured at hidden = 192 and kept at every shape point), so Inductor
multiplies by its reciprocal, as Helion does for a Python number.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm_train(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [rows, hidden] bf16; residual: [rows, hidden] fp32; weight, bias:
    # [hidden] fp32.
    rows, hidden = x.shape
    add_out = torch.empty([rows, hidden], dtype=torch.float32, device=x.device)
    normalized_out = torch.empty([rows, hidden], dtype=torch.float32, device=x.device)
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    rstd_out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        summed = residual[tile_r, :] + x[tile_r, :].to(torch.float32)
        add_out[tile_r, :] = summed
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1)
        centered = summed - mean[:, None]
        var = torch.mean(centered * centered, dim=-1)
        rstd = torch.rsqrt(var + 1e-6)
        normalized = centered * rstd[:, None]
        normalized_out[tile_r, :] = normalized
        y = normalized * weight[None, :] + bias[None, :]
        norm_out[tile_r, :] = y.to(norm_out.dtype)
        # div(rsqrt, 192): the graph's literal divisor, a reciprocal multiply
        # in Inductor too.
        rstd_out[tile_r] = rstd / 192
    return add_out, normalized_out, norm_out, rstd_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, residual, weight, bias = inputs[:4]
    out_shape = inputs[5]  # _shape_param_1 = [batch * seq, hidden]
    batch, seq, hidden = residual.shape
    add_out, normalized_out, norm_out, rstd_out = residual_add_layernorm_train(
        x, residual.view(batch * seq, hidden), weight, bias
    )
    return (
        add_out.view(batch, seq, hidden),
        normalized_out.view(batch, seq, hidden),
        norm_out.view(out_shape),
        rstd_out.view(batch, seq, 1),
    )
