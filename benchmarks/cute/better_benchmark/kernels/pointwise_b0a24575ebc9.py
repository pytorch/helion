"""better-benchmark repro pointwise_b0a24575ebc9 (timm inception_v3, adv_inception_v3).

The Inception-v3 stem before its first pooling: an inference BatchNorm + ReLU
of a channels-last bf16 ``[N, C, H, W]`` activation with per-channel bf16
statistics and affine parameters, followed by a max pooling without padding
(3x3 window, stride 2, dilation 1, floor mode at this shape point; the
pooling's offsets output is dead). The pooling reads each activation element
from several windows, so Inductor materializes the activation and emits two
pointwise kernels:

- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``:
  ``relu((x - mean) * rsqrt(var + 0.001) * weight + bias)``, computed in fp32
  (the graph's sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and its bf16
  rounding before the ReLU is not materialized) and stored in a bf16
  channels-last ``[N, C, H, W]`` buffer. Inductor indexes the dense storage
  as one flat dimension with ``channel = index % C`` for the four per-channel
  loads (five loads and one store per element). The port tiles the same
  storage as ``[N*H*W, C]`` (a view of the channels-last input) and loads the
  per-channel vectors once per channel tile.
- ``triton_poi_fused__low_memory_max_pool_with_offsets_add_convert_element_type_mul_relu_sub_unsqueeze_1``:
  each output element loads its ``kernel_h * kernel_w`` window of the bf16
  activation (nine loads) and folds the fp32 values with a NaN-propagating
  maximum in row-major window order before the bf16 store. Inductor indexes
  the dense channels-last ``[N, C, OH, OW]`` output as one flat dimension and
  recovers ``(n, oh, ow, c)`` from it with divisions and remainders; the port
  tiles those four dimensions of the same storage (``[N, H, W, C]`` and
  ``[N, OH, OW, C]`` views) and gathers the window with the same affine index
  math, ``(oh * stride_h + i, ow * stride_w + j)``.

The output is returned as the channels-last ``[N, C, OH, OW]`` view.
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
    # x: [N*H*W, C], a view of the channels-last activation; the rest [C].
    rows, channels = x.shape
    out = torch.empty([rows, channels], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y = x[tile_r, tile_c].to(torch.float32) - mean[tile_c].to(torch.float32)
        y = y * torch.rsqrt(var[tile_c].to(torch.float32) + 0.001)
        y = y * weight[tile_c].to(torch.float32) + bias[tile_c].to(torch.float32)
        out[tile_r, tile_c] = torch.relu(y).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def max_pool(
    x: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N, H, W, C], the storage-order view of the channels-last activation.
    n, height, width, channels = x.shape
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    out = torch.empty([n, out_h, out_w, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_h, tile_w, tile_c in hl.tile([n, out_h, out_w, channels]):
        row = (tile_h.index * stride_h)[:, None]
        col = (tile_w.index * stride_w)[None, :]
        best = x[tile_n, row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, kernel_h * kernel_w):
            i = offset // kernel_w
            j = offset % kernel_w
            value = x[tile_n, row + i, col + j, tile_c].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_n, tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    kernel_h, kernel_w = inputs[5]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[6]  # _shape_param_1 = stride
    n, c, h, w = x.shape
    act = batchnorm_relu(
        x.permute(0, 2, 3, 1).view(n * h * w, c), mean, var, weight, bias
    )
    out = max_pool(act.view(n, h, w, c), kernel_h, kernel_w, stride_h, stride_w)
    return (out.permute(0, 3, 1, 2),)
