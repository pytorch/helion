"""better-benchmark repro pointwise_88bffcefddc4 (timm mobilevit_s, tf_efficientnet_b0).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_div_exp_mul_neg_sub_unsqueeze_0``:
a BatchNorm in inference followed by SiLU,
``y = (x - mean) * rsqrt(var + 1e-5) * weight + bias`` and
``y / (exp(-y) + 1)``, of a channels-last bf16 ``[N, C, H, W]`` activation with
per-channel bf16 statistics and affine parameters, stored in bf16 in the same
layout. Inductor indexes the dense storage as one flat dimension with
``channel = index % C`` for the four per-channel loads (five loads and one
store per element) and computes everything in fp32: the graph's
sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and the graph's bf16 rounding of
the BatchNorm output before the SiLU is not materialized (no
``emulate_precision_casts``). The port tiles the same storage as
``[N*H*W, C]`` (a view of the channels-last input), loads the per-channel
vectors once per channel tile, and writes a ``[N*H*W, C]`` output the host
views back to the channels-last ``[N, C, H, W]``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn_silu(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # x: [N*H*W, C], a view of the channels-last activation; the rest [C].
    rows, channels = x.shape
    out = torch.empty([rows, channels], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y = x[tile_r, tile_c].to(torch.float32) - mean[tile_c].to(torch.float32)
        y = y * torch.rsqrt(var[tile_c].to(torch.float32) + 1e-05)
        y = y * weight[tile_c].to(torch.float32) + bias[tile_c].to(torch.float32)
        out[tile_r, tile_c] = (y / (torch.exp(-y) + 1)).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    n, c, h, w = x.shape
    rows = n * h * w
    out = bn_silu(x.permute(0, 2, 3, 1).view(rows, c), mean, var, weight, bias)
    return (out.view(n, h, w, c).permute(0, 3, 1, 2),)
