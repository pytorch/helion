"""better-benchmark repro pointwise_ef561a2def29 (torchbench resnet18 and others).

The ResNet stem after its first convolution: an inference BatchNorm + ReLU of
the bf16 activation ``[N, C, H, W]`` followed by a max pooling with padding
(3x3 window, stride 2, padding 1, dilation 1, floor mode at this shape point;
the pooling's offsets output is dead). The pooling reads each activation
element from several windows, so Inductor materializes the activation and
emits two pointwise kernels:

- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``:
  ``relu((x - mean) * rsqrt(var + 1e-05) * weight + bias)`` with the bf16
  per-channel statistics and affine parameters, computed in fp32 (the graph's
  sqrt/reciprocal becomes ``rsqrt``, and its bf16 rounding before the ReLU is
  not materialized) and stored in bf16. Inductor indexes the contiguous
  activation as one flat dimension and recovers the channel with a division
  and a remainder; the port tiles the same storage as ``[N, C, H * W]``.
- ``triton_poi_fused__low_memory_max_pool_with_offsets_add_convert_element_type_mul_relu_sub_unsqueeze_1``:
  each output element loads its ``kernel_h * kernel_w`` window of the bf16
  activation (nine loads), every load masked to the image with ``-inf`` in
  the padding, and folds the fp32 values with a NaN-propagating maximum in
  row-major window order before the bf16 store. Inductor indexes the
  contiguous output as one flat dimension and recovers ``(n * C + c, oh, ow)``
  from it; the port tiles those three dimensions of the same storage
  (``[N*C, H, W]`` and ``[N*C, OH, OW]`` views) and gathers the window with
  the same affine index math, ``(oh * stride_h - pad_h + i, ow * stride_w -
  pad_w + j)``.
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
    eps: hl.constexpr,
) -> torch.Tensor:
    # x: [N, C, H*W] bf16; mean, var, weight, bias: [C] bf16.
    batch, channels, plane = x.shape
    out = torch.empty([batch, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([batch, channels, plane]):
        centered = (
            x[tile_n, tile_c, tile_s].to(torch.float32)
            - mean[tile_c].to(torch.float32)[None, :, None]
        )
        rstd = torch.rsqrt(var[tile_c].to(torch.float32) + eps)
        y = centered * rstd[None, :, None]
        y = y * weight[tile_c].to(torch.float32)[None, :, None]
        y = y + bias[tile_c].to(torch.float32)[None, :, None]
        out[tile_n, tile_c, tile_s] = torch.relu(y).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def max_pool_padded(
    x: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
    pad_h: hl.constexpr,
    pad_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N*C, H, W], a view of the contiguous activation.
    planes, height, width = x.shape
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    out = torch.empty([planes, out_h, out_w], dtype=x.dtype, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        # The window's first row and column; negative in the padding.
        top = tile_h.index * stride_h - pad_h
        left = tile_w.index * stride_w - pad_w
        inside = ((top >= 0) & (top < height))[None, :, None] & (
            (left >= 0) & (left < width)
        )[None, None, :]
        value = hl.load(x, [tile_p, top[:, None], left[None, :]], extra_mask=inside).to(
            torch.float32
        )
        best = torch.where(inside, value, float("-inf"))
        for offset in hl.static_range(1, kernel_h * kernel_w):
            row = top + offset // kernel_w
            col = left + offset % kernel_w
            inside = ((row >= 0) & (row < height))[None, :, None] & (
                (col >= 0) & (col < width)
            )[None, None, :]
            value = hl.load(
                x, [tile_p, row[:, None], col[None, :]], extra_mask=inside
            ).to(torch.float32)
            best = torch.maximum(best, torch.where(inside, value, float("-inf")))
        out[tile_p, tile_h, tile_w] = best.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    kernel_h, kernel_w = inputs[5]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[6]  # _shape_param_1 = stride
    n, c, h, w = x.shape
    act = batchnorm_relu(x.view(n, c, h * w), mean, var, weight, bias, 1e-05)
    # The graph's padding is the literal [1, 1] (and its dilation [1, 1]).
    out = max_pool_padded(
        act.view(n * c, h, w), kernel_h, kernel_w, stride_h, stride_w, 1, 1
    )
    return (out.view(n, c, out.size(1), out.size(2)),)
