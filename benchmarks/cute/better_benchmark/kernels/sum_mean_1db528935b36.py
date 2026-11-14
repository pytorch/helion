"""better-benchmark repro sum_mean_1db528935b36 (timm/train/convnextv2_nano).

ConvNeXt-V2's exact (erf) GELU followed by Global Response Normalization on a
channels-last bf16 activation ``x`` of shape ``[N, C, H, W]`` with fp32
``beta``/``gamma`` ``[C]``: ``g = gelu(x)``, ``gx = sqrt(sum_{H,W} g^2)`` per
(n, c) (returned in fp32), ``denom = mean_C(gx) + 1e-6`` per n (returned in
fp32) and ``out = g + addcmul(beta, gamma, g * (gx / denom))`` in bf16,
channels-last like the input. Inductor emits three kernels, all on the input
viewed as ``[N, H*W, C]``:

- ``triton_red_fused_add_convert_element_type_erf_mul_pow_sum_0``: a looped
  reduction over H*W per (n, c); computes the GELU in fp32, sums its squares
  and stores the ``sqrt`` of the sum (``pow(sum, 0.5)``) as the fp32
  ``[N, C]`` output ``gx``.
- ``triton_red_fused_add_mean_1``: a looped reduction over C per n of ``gx``;
  stores ``sum / C + 1e-6`` as the fp32 ``[N]`` output ``denom``.
- ``triton_poi_fused_add_addcmul_convert_element_type_div_erf_mul_view_2``:
  recomputes the GELU and writes ``g + addcmul(beta, gamma, g * (gx / denom))``
  in bf16 (one fma in Inductor and on Triton; CuTe writes a multiply and add).

As in Inductor (no ``emulate_precision_casts``), the graph's bf16 round trip of
the GELU output stays in fp32; only the final output is rounded.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gelu_spatial_norm(x: torch.Tensor) -> torch.Tensor:
    # x: [N, H*W, C] bf16, a view of the channels-last input; returns
    # sqrt(sum_{H*W} gelu(x)^2) as fp32 [N, C].
    n, hw, c = x.shape
    gx = torch.empty([n, c], dtype=torch.float32, device=x.device)
    for tile_n, tile_c in hl.tile([n, c]):
        v = x[tile_n, :, tile_c].to(torch.float32)
        g = (v * 0.5) * (torch.erf(v * 0.7071067811865476) + 1.0)
        gx[tile_n, tile_c] = torch.sqrt(torch.sum(g * g, dim=1))
    return gx


@helion.kernel(static_shapes=True)
def channel_mean_eps(gx: torch.Tensor) -> torch.Tensor:
    # gx: [N, C] fp32; returns mean_C(gx) + 1e-6 as fp32 [N].
    n, c = gx.shape
    denom = torch.empty([n], dtype=torch.float32, device=gx.device)
    for tile_n in hl.tile(n):
        denom[tile_n] = torch.mean(gx[tile_n, :], dim=-1) + 1e-6
    return denom


@helion.kernel(static_shapes=True)
def gelu_grn_apply(
    x: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    gx: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    # x: [N, H*W, C] bf16; beta, gamma: [C] fp32; gx: [N, C] fp32;
    # denom: [N] fp32.
    n, hw, c = x.shape
    out = torch.empty([n, hw, c], dtype=x.dtype, device=x.device)
    for tile_n, tile_hw, tile_c in hl.tile([n, hw, c]):
        v = x[tile_n, tile_hw, tile_c].to(torch.float32)
        g = (v * 0.5) * (torch.erf(v * 0.7071067811865476) + 1.0)
        nx = gx[tile_n, tile_c] / denom[tile_n][:, None]
        y = torch.addcmul(beta[tile_c], gamma[tile_c], g * nx[:, None, :])
        out[tile_n, tile_hw, tile_c] = (g + y).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, beta, gamma = inputs[:3]
    n, c, h, w = x.shape
    x_flat = x.permute(0, 2, 3, 1).view(n, h * w, c)
    gx = gelu_spatial_norm(x_flat)
    denom = channel_mean_eps(gx)
    out = gelu_grn_apply(x_flat, beta, gamma, gx, denom)
    return (
        gx.view(n, c, 1, 1),
        denom.view(n, 1, 1, 1),
        out.view(n, h, w, c).permute(0, 3, 1, 2),
    )
