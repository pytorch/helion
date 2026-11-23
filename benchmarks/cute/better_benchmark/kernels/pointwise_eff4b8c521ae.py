"""better-benchmark repro pointwise_eff4b8c521ae (torchbench alexnet and vgg16).

Inductor emits one pointwise kernel,
``triton_poi_fused__low_memory_max_pool_with_offsets_relu_0``: a ReLU fused
into the max pooling that follows it (no padding, dilation 1, floor mode; the
pooling's offsets output is dead). Each output element loads its
``kernel_h * kernel_w`` window of the contiguous bf16 ``[N, C, H, W]``
activation (nine loads for the 3x3 window, stride 2, of this shape point),
applies the NaN-propagating ReLU to each element in fp32 and folds them with a
NaN-propagating maximum in row-major window order, then stores one bf16
element. Inductor indexes the contiguous output as one flat dimension and
recovers ``(n * C + c, oh, ow)`` from it with div/mod. The port tiles those
three dimensions of the same storage (``[N*C, H, W]`` and ``[N*C, OH, OW]``
views) and gathers the window with the same affine index math,
``(oh * stride_h + i, ow * stride_w + j)``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def relu_max_pool(
    x: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N*C, H, W], a view of the contiguous activation.
    planes, height, width = x.shape
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    out = torch.empty([planes, out_h, out_w], dtype=x.dtype, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        row = (tile_h.index * stride_h)[:, None]
        col = (tile_w.index * stride_w)[None, :]
        best = torch.relu(x[tile_p, row, col].to(torch.float32))
        for offset in hl.static_range(1, kernel_h * kernel_w):
            i = offset // kernel_w
            j = offset % kernel_w
            value = torch.relu(x[tile_p, row + i, col + j].to(torch.float32))
            best = torch.maximum(best, value)
        out[tile_p, tile_h, tile_w] = best.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    kernel_h, kernel_w = inputs[1]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[2]  # _shape_param_1 = stride
    n, c, h, w = arg0_1.shape
    out = relu_max_pool(
        arg0_1.view(n * c, h, w), kernel_h, kernel_w, stride_h, stride_w
    )
    return (out.view(n, c, out.size(1), out.size(2)),)
