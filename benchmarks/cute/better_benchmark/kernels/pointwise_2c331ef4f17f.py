"""better-benchmark repro pointwise_2c331ef4f17f (timm repvgg_a2, torchbench ResNets).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``: two
BatchNorm-inference branches (a residual block's main path and its shortcut,
or two RepVGG branches, each ``(x - mean) * rsqrt(var + 1e-5) * weight + bias``
with per-channel bf16 statistics and affine parameters) summed and passed
through a NaN-propagating ReLU, stored in bf16. At the ported shape points
(timm repvgg_a2) the two activations are channels-last ``[N, C, H, W]`` bf16
tensors with identical strides and the output keeps that layout. Inductor
indexes the dense storage as one flat dimension with ``channel = index % C``
for the eight per-channel loads (ten loads and one store per element),
computes everything in fp32 (the graph's sqrt/reciprocal/``* 1`` becomes
``rsqrt``, and the bf16 cast of each branch before the sum is not
materialized), and stores only the ReLU. The port tiles the same storage as
``[N*H*W, C]`` (a view of each channels-last input), loads the per-channel
vectors once per channel tile, and writes a ``[N*H*W, C]`` output the host
views back to the channels-last ``[N, C, H, W]``.

The repro's other shape points (torchbench LearningToPaint and ResNets) are
contiguous ``[N, C, H, W]`` activations, which Inductor indexes with
``channel = (index // (H*W)) % C``; ``forward()`` rejects them.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn2_add_relu(
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
) -> torch.Tensor:
    # x0, x1: [N*H*W, C], views of the channels-last activations; the rest [C].
    rows, channels = x0.shape
    out = torch.empty([rows, channels], dtype=x0.dtype, device=x0.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y0 = x0[tile_r, tile_c].to(torch.float32) - mean0[tile_c].to(torch.float32)
        y0 = y0 * torch.rsqrt(var0[tile_c].to(torch.float32) + 1e-05)
        y0 = y0 * weight0[tile_c].to(torch.float32) + bias0[tile_c].to(torch.float32)
        y1 = x1[tile_r, tile_c].to(torch.float32) - mean1[tile_c].to(torch.float32)
        y1 = y1 * torch.rsqrt(var1[tile_c].to(torch.float32) + 1e-05)
        y1 = y1 * weight1[tile_c].to(torch.float32) + bias1[tile_c].to(torch.float32)
        out[tile_r, tile_c] = torch.relu(y0 + y1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1 = inputs[:10]
    if not (
        x0.is_contiguous(memory_format=torch.channels_last)
        and x1.is_contiguous(memory_format=torch.channels_last)
    ):
        raise ValueError("this port handles channels-last activations")
    n, c, h, w = x0.shape
    rows = n * h * w
    out = bn2_add_relu(
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
    )
    return (out.view(n, h, w, c).permute(0, 3, 1, 2),)
