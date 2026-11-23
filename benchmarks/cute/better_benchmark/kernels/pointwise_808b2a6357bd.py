"""better-benchmark repro pointwise_808b2a6357bd (timm ghostnet_100).

The end of a GhostNet ghost module: the cheap branch's BatchNorm-inference +
ReLU of a channels-last bf16 activation
(``relu((x - mean) * rsqrt(var + 1e-05) * weight + bias)`` with per-channel
bf16 statistics and affine parameters) is concatenated along the channels
behind the primary branch's activation, which passes through unchanged.
Inductor emits one kernel,
``triton_poi_fused_add_cat_convert_element_type_mul_relu_sub_unsqueeze_0``: a
pointwise kernel over the dense channels-last ``[N, C0 + C1, H, W]`` output,
indexed as one flat dimension with ``channel = index % C`` and
``pixel = index // C``. The first ``C0`` channels are one load of the primary
activation masked to that channel range; the others are five loads (the cheap
activation and its four per-channel vectors) masked to theirs, computed in
fp32 (the graph's sqrt/reciprocal becomes ``rsqrt``, and the bf16 cast before
the ReLU is not materialized). The two are merged with a ``where`` and stored
in bf16, so an element reads either one activation or one activation and four
parameters.

The port tiles the same storage as ``[N*H*W, C]`` (the activations are
``[N*H*W, C_i]`` views) with the same masked loads, and returns the
channels-last ``[N, C, H, W]`` view of it.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cat_bn_relu(
    first: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # first: [N*H*W, C0], x: [N*H*W, C1], views of the channels-last
    # activations; mean, var, weight, bias: [C1] bf16.
    rows, first_channels = first.shape
    channels = first_channels + x.size(1)
    out = torch.empty([rows, channels], dtype=first.dtype, device=first.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        c = tile_c.index
        from_first = c < first_channels
        from_x = c >= first_channels
        cx = c - first_channels

        passed = hl.load(first, [tile_r, tile_c], extra_mask=from_first[None, :]).to(
            torch.float32
        )

        y = hl.load(x, [tile_r, cx], extra_mask=from_x[None, :]).to(torch.float32)
        y = y - hl.load(mean, [cx], extra_mask=from_x).to(torch.float32)
        y = y * torch.rsqrt(
            hl.load(var, [cx], extra_mask=from_x).to(torch.float32) + 1e-05
        )
        y = y * hl.load(weight, [cx], extra_mask=from_x).to(torch.float32)
        y = y + hl.load(bias, [cx], extra_mask=from_x).to(torch.float32)
        y = torch.relu(y)

        # cat([first, y], 1)
        merged = torch.where(from_first[None, :], passed, y)
        out[tile_r, tile_c] = merged.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias, first = inputs[:6]
    n, _, h, w = x.shape
    rows = n * h * w
    cat = cat_bn_relu(
        first.permute(0, 2, 3, 1).view(rows, first.size(1)),
        x.permute(0, 2, 3, 1).view(rows, x.size(1)),
        mean,
        var,
        weight,
        bias,
    )
    return (cat.view(n, h, w, cat.size(1)).permute(0, 3, 1, 2),)
