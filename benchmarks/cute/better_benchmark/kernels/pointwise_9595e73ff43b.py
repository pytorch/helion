"""better-benchmark repro pointwise_9595e73ff43b (timm mobilenetv2_100, torchbench mobilenet_v2).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_clamp_max_clamp_min_convert_element_type_mul_sub_unsqueeze_0``:
a BatchNorm in inference followed by ReLU6,
``y = (x - mean) * rsqrt(var + 1e-5) * weight + bias`` and
``clamp_max(clamp_min(y, 0), 6)``, of a channels-last bf16 ``[N, C, H, W]``
activation with per-channel bf16 statistics and affine parameters, stored in
bf16 in the same layout. Inductor indexes the dense storage as one flat
dimension with ``channel = index % C`` for the four per-channel loads (five
loads and one store per element) and computes everything in fp32: the graph's
sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and the graph's bf16 rounding of
the BatchNorm output before the clamps is not materialized (no
``emulate_precision_casts``; with bounds that bf16 represents exactly the
stored value is the same either way). The port tiles the same storage as
``[N*H*W, C]`` (a view of the input; the host views the output back), with
the per-channel vectors loaded once per channel tile. It handles the repro's
channels-last points (timm's); ``forward()`` rejects the contiguous ones.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn_relu6(
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
        y = torch.clamp_max(torch.clamp_min(y, 0.0), 6.0)
        out[tile_r, tile_c] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    if not x.is_contiguous(memory_format=torch.channels_last):
        # The repro's torchbench mobilenet_v2 shape points are contiguous
        # NCHW; timm's, the selected point among them, are channels-last.
        raise ValueError("this port handles channels-last activations")
    n, c, h, w = x.shape
    rows = n * h * w
    out = bn_relu6(x.permute(0, 2, 3, 1).view(rows, c), mean, var, weight, bias)
    return (out.view(n, h, w, c).permute(0, 3, 1, 2),)
