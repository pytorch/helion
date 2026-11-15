"""better-benchmark repro pointwise_9128d8745e42 (timm repvgg_a2 inference).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``: the
RepVGG block's three BatchNorm-inference branches (3x3 conv, 1x1 conv and
identity, each ``(x - mean) * rsqrt(var + 1e-5) * weight + bias`` with
per-channel bf16 statistics and affine parameters) summed and passed through a
NaN-propagating ReLU, stored in bf16. The three activations are channels-last
``[N, C, H, W]`` bf16 tensors with identical strides and the output keeps that
layout. Inductor indexes the dense storage as one flat dimension with
``channel = index % C`` for the twelve per-channel loads, computes everything
in fp32 (the graph's sqrt/reciprocal becomes ``rsqrt``, and the intermediate
bf16 casts of each branch and of the first sum are not materialized), and
stores only the ReLU. The port tiles the same storage as ``[N*H*W, C]`` (a
view of each channels-last input), loads the per-channel vectors once per
channel tile, and writes a ``[N*H*W, C]`` output the host views back to the
channels-last ``[N, C, H, W]``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def repvgg_bn3_add_relu(
    x0: torch.Tensor,
    mean0: torch.Tensor,
    var0: torch.Tensor,
    weight0: torch.Tensor,
    bias0: torch.Tensor,
    x1: torch.Tensor,
    mean1: torch.Tensor,
    var1: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    x2: torch.Tensor,
    mean2: torch.Tensor,
    var2: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    # x0, x1, x2: [N*H*W, C], views of the channels-last activations.
    rows, channels = x0.shape
    out = torch.empty([rows, channels], dtype=x0.dtype, device=x0.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y0 = x0[tile_r, tile_c].to(torch.float32) - mean0[tile_c].to(torch.float32)
        y0 = y0 * torch.rsqrt(var0[tile_c].to(torch.float32) + 1e-05)
        y0 = y0 * weight0[tile_c].to(torch.float32) + bias0[tile_c].to(torch.float32)
        y1 = x1[tile_r, tile_c].to(torch.float32) - mean1[tile_c].to(torch.float32)
        y1 = y1 * torch.rsqrt(var1[tile_c].to(torch.float32) + 1e-05)
        y1 = y1 * weight1[tile_c].to(torch.float32) + bias1[tile_c].to(torch.float32)
        y2 = x2[tile_r, tile_c].to(torch.float32) - mean2[tile_c].to(torch.float32)
        y2 = y2 * torch.rsqrt(var2[tile_c].to(torch.float32) + 1e-05)
        y2 = y2 * weight2[tile_c].to(torch.float32) + bias2[tile_c].to(torch.float32)
        out[tile_r, tile_c] = torch.relu(y0 + y1 + y2).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
    ) = inputs[:15]
    n, c, h, w = x0.shape
    rows = n * h * w
    out = repvgg_bn3_add_relu(
        x0.permute(0, 2, 3, 1).view(rows, c),
        mean0,
        var0,
        weight0,
        bias0,
        x1.permute(0, 2, 3, 1).view(rows, c),
        mean1,
        var1,
        weight1,
        bias1,
        x2.permute(0, 2, 3, 1).view(rows, c),
        mean2,
        var2,
        weight2,
        bias2,
    )
    return (out.view(n, h, w, c).permute(0, 3, 1, 2),)
