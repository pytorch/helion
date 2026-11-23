"""better-benchmark repro var_mean_787e1d544efe (timm beit_base_patch16_224 training).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_mul_rsqrt_sub_var_mean_view_0``:
a BEiT block's LayerScale residual add (a bf16 ``[batch * seq, hidden]``
projection output viewed to ``[batch, seq, hidden]``, times the fp32
per-channel LayerScale gamma, added to the fp32 stream of that shape) feeding
a LayerNorm over the hidden dim, in the training graph that also returns what
the backward needs. The product and the sum are computed in fp32 and the sum
is stored in fp32. Each row's population ``var_mean`` is computed in two
passes (``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``,
both true divisions by the count). The kernel stores four buffers: the fp32
sum, the fp32 normalized value ``(x - mean) * rsqrt(var + 1e-6)``, that value
times the fp32 weight plus the fp32 bias in bf16 (returned as a
``[batch * seq, hidden]`` view), and the per-row ``rsqrt(var + 1e-6) / 768``
in fp32 ``[batch, seq, 1]``. The divisor 768 is a literal of the graph, so
Inductor multiplies by its reciprocal, as Helion does for a Python number.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def layerscale_residual_add_layernorm_train(
    x: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [rows, hidden] bf16; residual: [rows, hidden] fp32; gamma, weight,
    # bias: [hidden] fp32.
    rows, hidden = x.shape
    add_out = torch.empty([rows, hidden], dtype=torch.float32, device=x.device)
    normalized_out = torch.empty([rows, hidden], dtype=torch.float32, device=x.device)
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    rstd_out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        scaled = gamma[None, :] * x[tile_r, :].to(torch.float32)
        summed = residual[tile_r, :] + scaled
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
        # div(rsqrt, 768): the graph's literal divisor, a reciprocal multiply
        # in Inductor too.
        rstd_out[tile_r] = rstd / 768
    return add_out, normalized_out, norm_out, rstd_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, gamma, residual, weight, bias = inputs[:5]
    out_shape = inputs[6]  # _shape_param_1 = [batch * seq, hidden]
    batch, seq, hidden = residual.shape
    add_out, normalized_out, norm_out, rstd_out = (
        layerscale_residual_add_layernorm_train(
            x, gamma, residual.view(batch * seq, hidden), weight, bias
        )
    )
    return (
        add_out.view(batch, seq, hidden),
        normalized_out.view(batch, seq, hidden),
        norm_out.view(out_shape),
        rstd_out.view(batch, seq, 1),
    )
