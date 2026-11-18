"""better-benchmark repro pointwise_498fa7e126b8 (timm tf_efficientnet_b0).

A BatchNorm in inference followed by SiLU and TensorFlow-style "same" padding
for the next stride-2 convolution (one zero row at the bottom and one zero
column at the right), of a channels-last bf16 ``[N, C, H, W]`` activation with
per-channel bf16 statistics and affine parameters. Inductor emits two
pointwise kernels, the BatchNorm over the image and the SiLU over the padded
output:

- ``triton_poi_fused_add_convert_element_type_mul_sub_unsqueeze_0``:
  ``y = (x - mean) * rsqrt(var + 0.001) * weight + bias``, computed in fp32
  and stored in an fp32 channels-last ``[N, C, H, W]`` buffer (the graph's
  sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and the graph's bf16 rounding of
  the BatchNorm output, a cast to bf16 and back, is not materialized: no
  ``emulate_precision_casts``). Inductor indexes the dense storage as one
  flat dimension with ``channel = index % C`` for the four per-channel loads
  (five loads and one store per element). The port tiles the same storage as
  ``[N*H*W, C]`` (a view of the channels-last input) and loads the
  per-channel vectors once per channel tile.
- ``triton_poi_fused_add_constant_pad_nd_convert_element_type_div_exp_neg_1``:
  a pointwise kernel over the dense channels-last ``[N, C, H + 1, W + 1]``
  bf16 output, indexed as one flat dimension from which Inductor recovers the
  batch, row and column with divisions and remainders. Each element loads the
  fp32 buffer masked to the rows and columns inside the image (the masked
  lanes read as zero), computes ``y / (exp(-y) + 1)`` in fp32, selects zero
  in the padding and stores bf16. The port tiles the ``[N, H + 1, W + 1, C]``
  storage view with the same masked load and select.

The output is returned as the channels-last ``[N, C, H + 1, W + 1]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def batchnorm_inference(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # x: [N*H*W, C], a view of the channels-last activation; the rest [C].
    rows, channels = x.shape
    out = torch.empty([rows, channels], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y = x[tile_r, tile_c].to(torch.float32) - mean[tile_c].to(torch.float32)
        y = y * torch.rsqrt(var[tile_c].to(torch.float32) + 0.001)
        y = y * weight[tile_c].to(torch.float32) + bias[tile_c].to(torch.float32)
        out[tile_r, tile_c] = y
    return out


@helion.kernel(static_shapes=True)
def silu_pad_bottom_right(
    y: torch.Tensor, pad_h: hl.constexpr, pad_w: hl.constexpr
) -> torch.Tensor:
    # y: [N, H, W, C] fp32, the storage-order view of the channels-last
    # BatchNorm output. Writes constant_pad_nd(silu(y).to(bf16),
    # [0, pad_w, 0, pad_h]) in the same storage order.
    n, height, width, channels = y.shape
    out_h = height + pad_h
    out_w = width + pad_w
    out = torch.empty(
        [n, out_h, out_w, channels], dtype=torch.bfloat16, device=y.device
    )
    for tile_n, tile_h, tile_w, tile_c in hl.tile([n, out_h, out_w, channels]):
        in_image = (tile_h.index < height)[None, :, None, None] & (
            tile_w.index < width
        )[None, None, :, None]
        v = hl.load(y, [tile_n, tile_h, tile_w, tile_c], extra_mask=in_image)
        silu = v / (torch.exp(-v) + 1)
        out[tile_n, tile_h, tile_w, tile_c] = torch.where(in_image, silu, 0.0).to(
            out.dtype
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    n, c, h, w = x.shape
    act = batchnorm_inference(
        x.permute(0, 2, 3, 1).view(n * h * w, c), mean, var, weight, bias
    )
    # constant_pad_nd(..., [0, 1, 0, 1], 0.0): one column right, one row below.
    out = silu_pad_bottom_right(act.view(n, h, w, c), 1, 1)
    return (out.permute(0, 3, 1, 2),)
