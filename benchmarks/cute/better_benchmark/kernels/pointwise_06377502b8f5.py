"""better-benchmark repro pointwise_06377502b8f5 (timm inception_v3, adv_inception_v3).

The tail of an Inception grid-reduction block and the start of the next
block: two branch outputs, each a BatchNorm-inference + ReLU of a
channels-last bf16 activation (``relu((x - mean) * rsqrt(var + 0.001) *
weight + bias)`` with per-channel bf16 statistics and affine parameters), and
a max pooling of the block's channels-last bf16 input (3x3 window, stride 2,
no padding, dilation 1, floor mode at this shape point; the pooling's offsets
output is dead) are concatenated along the channels, and the concatenation
also feeds the next block's 3x3 average pool (stride 1, padding 1, padded
cells counted). Inductor concatenates into a contiguous ``[N, C, H, W]``
buffer, each branch's kernel writing its own channel range of it, and then
copies that buffer into the channels-last layout of the returned
concatenation. It emits five pointwise kernels:

- ``triton_poi_fused__low_memory_max_pool_with_offsets_0``: a 2-D grid over
  the ``N*OH*OW`` output positions and the pooled branch's channels. Each
  element loads its ``kernel_h * kernel_w`` window of the channels-last
  activation (nine unmasked loads) and folds the fp32 values with a
  NaN-propagating maximum in row-major window order before the bf16 store
  into the last channels of the contiguous buffer. The port tiles the same
  four dimensions, ``(n, oh, ow, c)``, of the activation's storage-order
  ``[N, H, W, C]`` view and of the permuted view of that channel range, and
  gathers the window with the same affine index math,
  ``(oh * stride_h + i, ow * stride_w + j)``.
- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_1`` and
  ``..._2``: the BatchNorm + ReLU of each branch, a 2-D grid over the
  ``N*C_i`` planes and the ``H*W`` positions of a plane: one load of the
  channels-last activation and four per-channel loads, computed in fp32 (the
  graph's sqrt/reciprocal/``* 1`` becomes ``rsqrt``, and its bf16 rounding
  before the ReLU is not materialized) and stored in bf16 into the branch's
  channel range of the contiguous buffer. The port tiles ``[N, C_i, H*W]``
  views of both (the batch and the channels separately, since a channel
  range of the buffer is not dense). Each launch gets its own Helion kernel
  (same body) so each is tuned for its own shape.
- ``triton_poi_fused_add_cat_convert_element_type_mul_relu_sub_unsqueeze_3``:
  the layout copy, a 2-D grid over the ``N*C`` planes and the ``H*W``
  positions that loads the contiguous buffer (widened to fp32) and stores it
  in bf16 into the channels-last output. The port tiles the buffer's
  ``[N, C, H*W]`` view and writes the transposed ``[N, H*W, C]`` storage.
- ``triton_poi_fused_avg_pool2d_4``: a pointwise kernel over the
  channels-last storage that reads the concatenation at the nine window
  positions, each load masked to the image, sums them in fp32 in row-major
  window order and divides by the window's cell count
  ``(hend - hstart) * (wend - wstart)`` (computed from the output position,
  a true division; it is 9 everywhere for this pool) before the bf16 store.
  Inductor reuses the contiguous buffer, dead after the layout copy, as the
  channels-last storage of this output; the port passes the kernel the same
  reinterpretation of that buffer and tiles the ``[N, H, W, C]`` storage
  views.

Both outputs are returned as channels-last ``[N, C, H, W]`` views.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def max_pool(
    x: torch.Tensor,
    out: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N, H, W, C], the storage-order view of the channels-last activation;
    # out: [N, OH, OW, C], written in place: the permuted view of the pooled
    # branch's channels of the contiguous [N, C_total, OH, OW] concatenation.
    n, out_h, out_w, channels = out.shape
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


def _batchnorm_relu(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    # x: [N, C, H*W], a view of the channels-last activation; mean, var,
    # weight, bias: [C]; out: [N, C, H*W], written in place: the branch's
    # channels of the contiguous [N, C_total, H, W] concatenation.
    n, channels, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        y = x[tile_n, tile_c, tile_s].to(torch.float32)
        y = y - mean[tile_c].to(torch.float32)[None, :, None]
        y = y * torch.rsqrt(var[tile_c].to(torch.float32) + 0.001)[None, :, None]
        y = y * weight[tile_c].to(torch.float32)[None, :, None]
        y = y + bias[tile_c].to(torch.float32)[None, :, None]
        out[tile_n, tile_c, tile_s] = torch.relu(y).to(out.dtype)
    return out


batchnorm_relu_0 = helion.kernel(static_shapes=True)(_batchnorm_relu)
batchnorm_relu_1 = helion.kernel(static_shapes=True)(_batchnorm_relu)


@helion.kernel(static_shapes=True)
def cat_to_channels_last(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, H*W], the contiguous concatenation. Returns its channels-last
    # copy in storage order, [N, H*W, C].
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        v = x[tile_n, tile_c, tile_s].to(torch.float32)
        out[tile_n, tile_s, tile_c] = v.permute(0, 2, 1).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def avg_pool_3x3(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # x: [N, H, W, C], the storage-order view of the channels-last input;
    # out: [N, H, W, C], written in place: the storage of the contiguous
    # concatenation, reused in the channels-last order.
    n, height, width, channels = x.shape
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
    x_pool = inputs[0]
    mean0, x0, var0, weight0, bias0 = inputs[1:6]
    mean1, x1, var1, weight1, bias1 = inputs[6:11]
    kernel_h, kernel_w = inputs[11]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[12]  # _shape_param_1 = stride
    if not all(
        t.is_contiguous(memory_format=torch.channels_last) for t in (x_pool, x0, x1)
    ):
        raise ValueError("this port handles channels-last activations")
    n, c0, h, w = x0.shape
    c1 = x1.size(1)
    channels = c0 + c1 + x_pool.size(1)
    plane = h * w
    # The contiguous concatenation cat([relu, relu_1, getitem], 1); each
    # branch's kernel writes its own channels.
    staging = torch.empty([n, channels, h, w], dtype=x0.dtype, device=x0.device)
    # The graph's max-pool padding is the literal [0, 0] (dilation [1, 1]).
    max_pool(
        x_pool.permute(0, 2, 3, 1),
        staging[:, c0 + c1 :].permute(0, 2, 3, 1),
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
    )
    batchnorm_relu_0(
        x0.view(n, c0, plane),
        mean0,
        var0,
        weight0,
        bias0,
        staging[:, :c0].view(n, c0, plane),
    )
    batchnorm_relu_1(
        x1.view(n, c1, plane),
        mean1,
        var1,
        weight1,
        bias1,
        staging[:, c0 : c0 + c1].view(n, c1, plane),
    )
    cat = cat_to_channels_last(staging.view(n, channels, plane))
    cat = cat.view(n, h, w, channels)
    # avg_pool2d(cat, [3, 3], [1, 1], [1, 1]): graph literals. Its output
    # takes over the storage of the contiguous concatenation.
    pooled = avg_pool_3x3(cat, staging.view(n, h, w, channels))
    return cat.permute(0, 3, 1, 2), pooled.permute(0, 3, 1, 2)
