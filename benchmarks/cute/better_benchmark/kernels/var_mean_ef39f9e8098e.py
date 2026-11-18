"""better-benchmark repro var_mean_ef39f9e8098e (timm/train/vit_base_patch14_dinov2.lvd142m).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_mul_rsqrt_sub_var_mean_view_0``:
the training forward of the DINOv2 block's LayerScale residual add (a bf16
``[batch * seq, hidden]`` projection output times the fp32 per-channel
LayerScale gamma, added to the fp32 ``[batch, seq, hidden]`` stream) followed
by a LayerNorm over the hidden dim. The product and the sum stay in fp32 and
the sum is returned in fp32. Each row's population ``var_mean`` is computed in
two passes (``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``).
The kernel stores four buffers: the fp32 sum, the fp32 normalized rows
``(x - mean) * rsqrt(var + 1e-6)`` that autograd saves, the affine output
``normalized * weight + bias`` in bf16 (returned as a ``[batch * seq, hidden]``
view), and the per-row ``rsqrt(var + 1e-6) / 768`` (the graph's literal
divisor, a multiply by its reciprocal in Inductor) that the backward uses.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def layerscale_residual_add_layernorm_train(
    branch: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, hidden = branch.shape
    add_out = torch.empty([rows, hidden], dtype=torch.float32, device=branch.device)
    normalized_out = torch.empty(
        [rows, hidden], dtype=torch.float32, device=branch.device
    )
    norm_out = torch.empty([rows, hidden], dtype=branch.dtype, device=branch.device)
    rstd_div_out = torch.empty([rows], dtype=torch.float32, device=branch.device)
    for tile_r in hl.tile(rows):
        scaled = branch[tile_r, :].to(torch.float32) * gamma[None, :].to(torch.float32)
        summed = residual[tile_r, :].to(torch.float32) + scaled
        add_out[tile_r, :] = summed
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1)
        centered = summed - mean[:, None]
        var = torch.mean(centered * centered, dim=-1)
        rstd = torch.rsqrt(var + 1e-6)
        normalized = centered * rstd[:, None]
        normalized_out[tile_r, :] = normalized
        y = normalized * weight[None, :].to(torch.float32) + bias[None, :].to(
            torch.float32
        )
        norm_out[tile_r, :] = y.to(norm_out.dtype)
        # The graph's div(rsqrt, 768): a literal divisor, which Inductor also
        # turns into a multiply by its reciprocal.
        rstd_div_out[tile_r] = rstd / 768
    return add_out, normalized_out, norm_out, rstd_div_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    branch, gamma, residual, weight, bias = inputs[:5]
    out_shape = inputs[6]  # _shape_param_1 = [batch * seq, hidden]
    batch, seq, hidden = residual.shape
    add_out, normalized_out, norm_out, rstd_div_out = (
        layerscale_residual_add_layernorm_train(
            branch, gamma, residual.view(batch * seq, hidden), weight, bias
        )
    )
    return (
        add_out.view(batch, seq, hidden),
        normalized_out.view(batch, seq, hidden),
        norm_out.view(out_shape),
        rstd_div_out.view(batch, seq, 1),
    )
