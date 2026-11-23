"""better-benchmark repro pointwise_b71f0d447f2e (timm inception_v3, adv_inception_v3).

The tail of an Inception block: the four branch outputs, each a
BatchNorm-inference + ReLU of a channels-last bf16 activation
(``relu((x - mean) * rsqrt(var + 0.001) * weight + bias)`` with per-channel
bf16 statistics and affine parameters), are concatenated along the channels,
and the concatenation also feeds the next block's 3x3 average pool (stride 1,
padding 1, padded cells counted). Inductor emits two kernels:

- ``triton_poi_fused_add_cat_convert_element_type_mul_relu_sub_unsqueeze_0``:
  a pointwise kernel over the dense channels-last ``[N, C0+C1+C2+C3, H, W]``
  output, indexed as one flat dimension with ``channel = index % C``. Each
  branch is five loads (the activation and its four per-channel vectors)
  masked to the branch's channel range, so an element reads one activation
  and four parameters; the branches are computed in fp32 (the graph's
  sqrt/reciprocal becomes ``rsqrt``, and the bf16 cast before the ReLU is
  not materialized), merged with a ``where`` chain and stored in bf16. The
  port tiles the same storage as ``[N*H*W, C]`` (the activations are
  ``[N*H*W, C_i]`` views) with the same masked loads.
- ``triton_poi_fused_avg_pool2d_1``: a pointwise kernel over the same
  storage that reads the concatenation at the nine window positions, each
  load masked to the image, sums them in fp32 in row-major window order and
  divides by the window's cell count ``(hend - hstart) * (wend - wstart)``
  (computed from the output position, a true division; it is 9 everywhere
  for this pool) before the bf16 store. The port tiles the ``[N, H, W, C]``
  storage view.

Both outputs are returned as channels-last ``[N, C, H, W]`` views.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn_relu_cat4(
    x0: torch.Tensor,
    mean0: torch.Tensor,
    var0: torch.Tensor,
    weight0: torch.Tensor,
    bias0: torch.Tensor,
    x1: torch.Tensor,
    mean1: torch.Tensor,
    var1: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    x2: torch.Tensor,
    mean2: torch.Tensor,
    var2: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    x3: torch.Tensor,
    mean3: torch.Tensor,
    var3: torch.Tensor,
    weight3: torch.Tensor,
    bias3: torch.Tensor,
) -> torch.Tensor:
    # x0..x3: [N*H*W, C_i], views of the channels-last activations.
    rows = x0.size(0)
    end0 = x0.size(1)
    end1 = end0 + x1.size(1)
    end2 = end1 + x2.size(1)
    channels = end2 + x3.size(1)
    out = torch.empty([rows, channels], dtype=x0.dtype, device=x0.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        c = tile_c.index
        in0 = c < end0
        in1 = (c >= end0) & (c < end1)
        in2 = (c >= end1) & (c < end2)
        in3 = c >= end2
        c1 = c - end0
        c2 = c - end1
        c3 = c - end2

        y0 = hl.load(x0, [tile_r, tile_c], extra_mask=in0[None, :]).to(torch.float32)
        y0 = y0 - hl.load(mean0, [tile_c], extra_mask=in0).to(torch.float32)
        y0 = y0 * torch.rsqrt(
            hl.load(var0, [tile_c], extra_mask=in0).to(torch.float32) + 0.001
        )
        y0 = y0 * hl.load(weight0, [tile_c], extra_mask=in0).to(torch.float32)
        y0 = y0 + hl.load(bias0, [tile_c], extra_mask=in0).to(torch.float32)
        y0 = torch.relu(y0)

        y1 = hl.load(x1, [tile_r, c1], extra_mask=in1[None, :]).to(torch.float32)
        y1 = y1 - hl.load(mean1, [c1], extra_mask=in1).to(torch.float32)
        y1 = y1 * torch.rsqrt(
            hl.load(var1, [c1], extra_mask=in1).to(torch.float32) + 0.001
        )
        y1 = y1 * hl.load(weight1, [c1], extra_mask=in1).to(torch.float32)
        y1 = y1 + hl.load(bias1, [c1], extra_mask=in1).to(torch.float32)
        y1 = torch.relu(y1)

        y2 = hl.load(x2, [tile_r, c2], extra_mask=in2[None, :]).to(torch.float32)
        y2 = y2 - hl.load(mean2, [c2], extra_mask=in2).to(torch.float32)
        y2 = y2 * torch.rsqrt(
            hl.load(var2, [c2], extra_mask=in2).to(torch.float32) + 0.001
        )
        y2 = y2 * hl.load(weight2, [c2], extra_mask=in2).to(torch.float32)
        y2 = y2 + hl.load(bias2, [c2], extra_mask=in2).to(torch.float32)
        y2 = torch.relu(y2)

        y3 = hl.load(x3, [tile_r, c3], extra_mask=in3[None, :]).to(torch.float32)
        y3 = y3 - hl.load(mean3, [c3], extra_mask=in3).to(torch.float32)
        y3 = y3 * torch.rsqrt(
            hl.load(var3, [c3], extra_mask=in3).to(torch.float32) + 0.001
        )
        y3 = y3 * hl.load(weight3, [c3], extra_mask=in3).to(torch.float32)
        y3 = y3 + hl.load(bias3, [c3], extra_mask=in3).to(torch.float32)
        y3 = torch.relu(y3)

        # cat([y0, y1, y2, y3], 1)
        merged = torch.where(
            in0[None, :],
            y0,
            torch.where(in1[None, :], y1, torch.where(in2[None, :], y2, y3)),
        )
        out[tile_r, tile_c] = merged.to(out.dtype)
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
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
    ) = inputs[:20]
    n, _, h, w = x0.shape
    rows = n * h * w
    cat = bn_relu_cat4(
        x0.permute(0, 2, 3, 1).view(rows, x0.size(1)),
        mean0,
        var0,
        weight0,
        bias0,
        x1.permute(0, 2, 3, 1).view(rows, x1.size(1)),
        mean1,
        var1,
        weight1,
        bias1,
        x2.permute(0, 2, 3, 1).view(rows, x2.size(1)),
        mean2,
        var2,
        weight2,
        bias2,
        x3.permute(0, 2, 3, 1).view(rows, x3.size(1)),
        mean3,
        var3,
        weight3,
        bias3,
    )
    cat = cat.view(n, h, w, cat.size(1))
    pooled = avg_pool_3x3(cat)
    return cat.permute(0, 3, 1, 2), pooled.permute(0, 3, 1, 2)
