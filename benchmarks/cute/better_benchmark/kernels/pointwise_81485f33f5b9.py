"""better-benchmark repro pointwise_81485f33f5b9 (torchbench resnet152, resnet50 and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``: a
ResNet block's residual join in inference, ``relu(((x - mean) *
rsqrt(var + 1e-5) * weight + bias) + residual)``, a BatchNorm with per-channel
bf16 statistics and affine parameters applied to a bf16 NCHW activation, plus
the bf16 shortcut activation, through a NaN-propagating ReLU, stored in bf16.
Both activations are contiguous at every shape point, so Inductor indexes them
and the output as one flat dimension (two loads and one store per element) and
gathers the four per-channel operands with the channel ``(i // (H*W)) % C``.
It computes everything in fp32: the graph's sqrt/reciprocal/``* 1`` becomes
``rsqrt``, and the graph's bf16 rounding of the BatchNorm output before the
add is not materialized (no ``emulate_precision_casts``). The port tiles the
same flat views with the same index math.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn_add_relu(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    # x, residual: [N, C, H, W] contiguous bf16; the rest [C] bf16.
    batch, channels, height, width = x.shape
    hw = height * width
    x_flat = x.view(-1)
    residual_flat = residual.view(-1)
    numel = x_flat.size(0)
    out = torch.empty([numel], dtype=x.dtype, device=x.device)
    for tile in hl.tile(numel):
        c = (tile.index // hw) % channels
        y = x_flat[tile].to(torch.float32) - mean[c].to(torch.float32)
        y = y * torch.rsqrt(var[c].to(torch.float32) + 1e-05)
        y = y * weight[c].to(torch.float32) + bias[c].to(torch.float32)
        y = y + residual_flat[tile].to(torch.float32)
        out[tile] = torch.relu(y).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias, residual = inputs[:6]
    out = bn_add_relu(x, mean, var, weight, bias, residual)
    return (out.view(x.shape),)
