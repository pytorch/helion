"""better-benchmark repro pointwise_7899d74e0838 (torchbench pytorch_unet and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``: an
inference BatchNorm + ReLU of a bf16 ``[N, C, H, W]`` activation with bf16
per-channel statistics and affine parameters,
``relu((x - mean) * rsqrt(var + 0.001) * weight + bias)``, computed in fp32
and stored in bf16 (the graph's sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and
the graph's bf16 rounding before the ReLU is not materialized: the ReLU runs
in fp32 and the store rounds, which gives the same values).

At the ported shape point (pytorch_unet, ``[1, 64, 640, 959]``) the activation
is contiguous NCHW. Inductor indexes its dense storage as one flat dimension
with ``channel = index // (H*W)`` for the four per-channel loads (five loads
and one store per element). The port tiles the same storage as
``[N, C, H*W]`` (a free view of the contiguous input) and loads the
per-channel vectors once per channel tile. The repro's channels-last shape
points have another storage order (``channel = index % C`` in Inductor) and
are rejected in ``forward()``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def batchnorm_relu(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # x: [N, C, H*W] bf16; mean, var, weight, bias: [C] bf16.
    batch, channels, plane = x.shape
    out = torch.empty([batch, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([batch, channels, plane]):
        centered = (
            x[tile_n, tile_c, tile_s].to(torch.float32)
            - mean[tile_c].to(torch.float32)[None, :, None]
        )
        rstd = torch.rsqrt(var[tile_c].to(torch.float32) + 0.001)
        y = centered * rstd[None, :, None]
        y = y * weight[tile_c].to(torch.float32)[None, :, None]
        y = y + bias[tile_c].to(torch.float32)[None, :, None]
        out[tile_n, tile_c, tile_s] = torch.relu(y).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    if not x.is_contiguous():
        # 26 of the repro's shape points are channels-last, the first one
        # among them.
        raise ValueError("this port handles contiguous NCHW activations")
    batch, channels, height, width = x.shape
    out = batchnorm_relu(
        x.view(batch, channels, height * width), mean, var, weight, bias
    )
    return (out.view(batch, channels, height, width),)
