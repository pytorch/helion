"""better-benchmark repro sum_mean_6150664fe0bf (timm convnextv2_nano).

ConvNeXt-V2's exact (erf) GELU followed by Global Response Normalization on a
channels-last bf16 activation ``x`` of shape ``[N, C, H, W]``:
``g = gelu(x)``, ``gx = sqrt(sum_{H,W} g^2)`` per (n, c),
``nx = gx / (mean_C(gx) + 1e-6)`` and ``out = g + addcmul(beta, gamma, g * nx)``
in bf16, channels-last like the input. Inductor emits three kernels, all on
the input viewed as ``[N, H*W, C]``:

- ``triton_red_fused_add_convert_element_type_erf_mul_pow_sum_0``: a looped
  reduction over H*W per (n, c); computes the GELU in fp32 and sums its
  squares into an fp32 ``[N, C]`` workspace.
- ``triton_red_fused_convert_element_type_mean_pow_1``: a looped reduction
  over C per n of ``sqrt`` of the workspace; stores the plain sum (the
  mean's numerator) in an fp32 ``[N]`` buffer.
- ``triton_poi_fused_add_addcmul_convert_element_type_div_erf_mean_mul_pow_view_2``:
  recomputes the GELU, finishes the mean (a true division of that sum by C)
  and writes ``g + addcmul(beta, gamma, g * (sqrt(sum_sq) / (mean + 1e-6)))``
  in bf16 (Inductor and Helion's Triton backend lower the addcmul to an fma;
  the CuTe backend currently renders ``prims.fma`` as a multiply and an add).

As in Inductor (no ``emulate_precision_casts``), the graph's bf16 round trips
of the GELU output, of ``gx``, of the mean and of the products stay in fp32;
only the output is rounded.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gelu_sum_squares(x: torch.Tensor) -> torch.Tensor:
    # x: [N, H*W, C] bf16, a view of the channels-last input.
    n, hw, c = x.shape
    sum_sq = torch.empty([n, c], dtype=torch.float32, device=x.device)
    for tile_n, tile_c in hl.tile([n, c]):
        v = x[tile_n, :, tile_c].to(torch.float32)
        g = (v * 0.5) * (torch.erf(v * 0.7071067811865476) + 1.0)
        sum_sq[tile_n, tile_c] = torch.sum(g * g, dim=1)
    return sum_sq


@helion.kernel(static_shapes=True)
def sqrt_channel_sum(sum_sq: torch.Tensor) -> torch.Tensor:
    # sum_sq: [N, C] fp32; returns sum_C(sqrt(sum_sq)) as fp32 [N].
    n, c = sum_sq.shape
    gx_sum = torch.empty([n], dtype=torch.float32, device=sum_sq.device)
    for tile_n in hl.tile(n):
        gx_sum[tile_n] = torch.sum(torch.sqrt(sum_sq[tile_n, :]), dim=-1)
    return gx_sum


@helion.kernel(static_shapes=True)
def gelu_grn_apply(
    x: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    sum_sq: torch.Tensor,
    gx_sum: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    # x: [N, H*W, C] bf16; beta, gamma: [C] bf16; sum_sq: [N, C] fp32;
    # gx_sum: [N] fp32. ``channels`` (C, the mean's count) stays a runtime
    # scalar: a constant divisor would become a reciprocal multiply, while
    # Inductor truly divides the channel sum by C in this kernel.
    n, hw, c = x.shape
    out = torch.empty([n, hw, c], dtype=x.dtype, device=x.device)
    for tile_n, tile_hw, tile_c in hl.tile([n, hw, c]):
        v = x[tile_n, tile_hw, tile_c].to(torch.float32)
        g = (v * 0.5) * (torch.erf(v * 0.7071067811865476) + 1.0)
        gx_mean = gx_sum[tile_n] / channels
        nx = torch.sqrt(sum_sq[tile_n, tile_c]) / (gx_mean[:, None] + 1e-6)
        y = torch.addcmul(
            beta[tile_c].to(torch.float32),
            gamma[tile_c].to(torch.float32),
            g * nx[:, None, :],
        )
        out[tile_n, tile_hw, tile_c] = (g + y).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    beta, gamma, x = inputs[:3]
    n, c, h, w = x.shape
    x_flat = x.permute(0, 2, 3, 1).view(n, h * w, c)
    sum_sq = gelu_sum_squares(x_flat)
    gx_sum = sqrt_channel_sum(sum_sq)
    out = gelu_grn_apply(x_flat, beta, gamma, sum_sq, gx_sum, c)
    return (out.view(n, h, w, c).permute(0, 3, 1, 2),)
