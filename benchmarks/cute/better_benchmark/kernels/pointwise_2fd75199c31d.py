"""better-benchmark repro pointwise_2fd75199c31d (timm inception_v3, adv_inception_v3).

The end of the Inception stem and the start of the first Inception block: an
inference BatchNorm + ReLU of a channels-last bf16 ``[N, C, H, W]`` activation,
a max pooling without padding (3x3 window, stride 2, dilation 1, floor mode at
this shape point; the pooling's offsets output is dead), and the next block's
3x3 average pool of the pooled activation (stride 1, padding 1, padded cells
counted). The max pooling reads each activation element from several windows
and its output has two users (it is returned and averaged), so Inductor
materializes both and emits three pointwise kernels:

- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``:
  ``relu((x - mean) * rsqrt(var + 0.001) * weight + bias)`` with the bf16
  per-channel statistics and affine parameters, computed in fp32 (the graph's
  sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and its bf16 rounding before the
  ReLU is not materialized) and stored in a bf16 channels-last buffer.
  Inductor indexes the dense storage as one flat dimension with
  ``channel = index % C`` for the four per-channel loads (five loads and one
  store per element). The port tiles the same storage as ``[N*H*W, C]`` (a
  view of the channels-last input).
- ``triton_poi_fused__low_memory_max_pool_with_offsets_add_convert_element_type_mul_relu_sub_unsqueeze_1``:
  each element of the dense channels-last ``[N, C, OH, OW]`` output loads its
  ``kernel_h * kernel_w`` window of the bf16 activation (nine unmasked
  loads) and folds the fp32 values with a NaN-propagating maximum in
  row-major window order before the bf16 store. Inductor indexes the output
  as one flat dimension and recovers ``(n, oh, ow, c)`` from it with
  divisions and remainders; the port tiles those four dimensions of the same
  storage (``[N, H, W, C]`` and ``[N, OH, OW, C]`` views) and gathers the
  window with the same affine index math,
  ``(oh * stride_h + i, ow * stride_w + j)``.
- ``triton_poi_fused_avg_pool2d_2``: a pointwise kernel over the same
  ``[N, OH, OW, C]`` storage that reads the pooled activation at the nine
  window positions, each load masked to the image, sums them in fp32 in
  row-major window order and divides by the window's cell count
  ``(hend - hstart) * (wend - wstart)`` (computed from the output position,
  a true division; it is 9 everywhere for this pool) before the bf16 store.

Both outputs are returned as channels-last ``[N, C, OH, OW]`` views.
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
        # The window's first row and column.
        row = tile_h.index * stride_h
        col = tile_w.index * stride_w
        best = hl.load(x, [tile_n, row, col, tile_c]).to(torch.float32)
        for offset in hl.static_range(1, kernel_h * kernel_w):
            i = offset // kernel_w
            j = offset % kernel_w
            value = hl.load(x, [tile_n, row + i, col + j, tile_c]).to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_n, tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def avg_pool_3x3(x: torch.Tensor) -> torch.Tensor:
    # x: [N, H, W, C], the storage-order view of the channels-last input.
    n, height, width, channels = x.shape
    out = torch.empty([n, height, width, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_h, tile_w, tile_c in hl.tile([n, height, width, channels]):
        h = tile_h.index
        w = tile_w.index
        # Which neighbours of each output position lie inside the image.
        above = (h >= 1)[None, :, None, None]
        below = (h + 1 < height)[None, :, None, None]
        left = (w >= 1)[None, None, :, None]
        right = (w + 1 < width)[None, None, :, None]

        # The 3x3 window in row-major order, v<row><col>.
        v00 = hl.load(x, [tile_n, h - 1, w - 1, tile_c], extra_mask=above & left)
        v01 = hl.load(x, [tile_n, h - 1, tile_w, tile_c], extra_mask=above)
        v02 = hl.load(x, [tile_n, h - 1, w + 1, tile_c], extra_mask=above & right)
        v10 = hl.load(x, [tile_n, tile_h, w - 1, tile_c], extra_mask=left)
        v11 = x[tile_n, tile_h, tile_w, tile_c]
        v12 = hl.load(x, [tile_n, tile_h, w + 1, tile_c], extra_mask=right)
        v20 = hl.load(x, [tile_n, h + 1, w - 1, tile_c], extra_mask=below & left)
        v21 = hl.load(x, [tile_n, h + 1, tile_w, tile_c], extra_mask=below)
        v22 = hl.load(x, [tile_n, h + 1, w + 1, tile_c], extra_mask=below & right)
        total = (
            v00.to(torch.float32)
            + v01.to(torch.float32)
            + v02.to(torch.float32)
            + v10.to(torch.float32)
            + v11.to(torch.float32)
            + v12.to(torch.float32)
            + v20.to(torch.float32)
            + v21.to(torch.float32)
            + v22.to(torch.float32)
        )

        # count_include_pad: the window [h - 1, h + 2) x [w - 1, w + 2)
        # clipped to the padded image [-1, H + 1) x [-1, W + 1).
        h_cells = torch.clamp(h + 2, max=height + 1) - (h - 1)
        w_cells = torch.clamp(w + 2, max=width + 1) - (w - 1)
        cells = (h_cells[:, None] * w_cells[None, :]).to(torch.float32)
        out[tile_n, tile_h, tile_w, tile_c] = (total / cells[None, :, :, None]).to(
            out.dtype
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias = inputs[:5]
    kernel_h, kernel_w = inputs[5]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[6]  # _shape_param_1 = stride
    n, c, h, w = x.shape
    act = batchnorm_relu(
        x.permute(0, 2, 3, 1).view(n * h * w, c), mean, var, weight, bias
    )
    # The graph's max-pool padding is the literal [0, 0] (dilation [1, 1]).
    pooled = max_pool(act.view(n, h, w, c), kernel_h, kernel_w, stride_h, stride_w)
    # avg_pool2d(pooled, [3, 3], [1, 1], [1, 1]): graph literals.
    averaged = avg_pool_3x3(pooled)
    return pooled.permute(0, 3, 1, 2), averaged.permute(0, 3, 1, 2)
