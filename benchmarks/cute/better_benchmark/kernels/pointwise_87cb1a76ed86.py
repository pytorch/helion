"""better-benchmark repro pointwise_87cb1a76ed86 (torchbench pytorch_unet).

One U-Net decoder step: the inference BatchNorm + ReLU that ends a
``DoubleConv``, the 2x bilinear upsample (``align_corners=True``) of its
result, and the concatenation of the skip connection with the upsampled
activation, zero-padded by one column to the skip's width. Inductor emits
three pointwise kernels:

- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_0``:
  ``relu((x - mean) * rsqrt(var + 1e-05) * weight + bias)`` of the bf16
  activation ``[N, C, H, W]`` with the bf16 per-channel statistics, computed
  in fp32 and stored in fp32 (as in Inductor, the graph's bf16 rounding before
  the ReLU stays in fp32).
- ``triton_poi_fused__unsafe_index_add_clamp_max_clamp_min_convert_element_type_iota_mul_sub_view_1``:
  the bilinear upsample to ``[N, C, 320, 478]`` in fp32. Each output element
  computes its source row and column from its own position
  (``clamp_min(float(iota) * scale, 0)``, truncated; the neighbour is
  ``min(index + 1, size - 1)``), gathers the four neighbours from the fp32
  activation and interpolates along the width and then the height with
  weights clamped to ``[0, 1]``.
- ``triton_poi_fused_cat_constant_pad_nd_convert_element_type_2``: writes the
  bf16 ``[N, 2C, 320, 479]`` output: the first C channels are a masked load
  of the bf16 skip connection, the others a masked load of the fp32 upsample
  (cast to bf16) for the first 478 columns and zero in the padded column.

The port tiles each kernel's own dims (Inductor indexes them as one flat
dimension and recovers the channel, row and column with divisions and
remainders): the BatchNorm over ``[N, C, H * W]``, the upsample over
``[N * C, 320, 478]`` and the concatenation over ``[N, 2C, 320 * 479]``, with
the skip connection and the output viewed with their rows merged. The
views are free, since every buffer is contiguous.

The concatenation takes the row and the column of a plane position as
Inductor's kernel takes them of its flat index
(``x1 = (xindex // 479) % 320``, ``x0 = xindex % 479``), for the upsample's
load alone: that buffer's rows are one column shorter than the output's.
Like Inductor's blocks, its tiles run across the ends of a plane's rows;
tiles over ``[N, 2C, 320, 479]`` would end with each row, an odd length.
Eager reference mode (``HELION_INTERPRET=1``) runs this kernel as well: its
masked ``hl.load`` lays index tensors beside a tile out as compiled code does.
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
    out = torch.empty([batch, channels, plane], dtype=torch.float32, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([batch, channels, plane]):
        centered = (
            x[tile_n, tile_c, tile_s].to(torch.float32)
            - mean[tile_c].to(torch.float32)[None, :, None]
        )
        rstd = torch.rsqrt(var[tile_c].to(torch.float32) + eps)
        y = centered * rstd[None, :, None]
        y = y * weight[tile_c].to(torch.float32)[None, :, None]
        y = y + bias[tile_c].to(torch.float32)[None, :, None]
        out[tile_n, tile_c, tile_s] = torch.relu(y)
    return out


@helion.kernel(static_shapes=True)
def upsample_bilinear(
    x: torch.Tensor,
    out_h: hl.constexpr,
    out_w: hl.constexpr,
    scale_h: hl.constexpr,
    scale_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N*C, H, W] fp32; the output is [N*C, out_h, out_w] fp32.
    planes, in_h, in_w = x.shape
    out = torch.empty([planes, out_h, out_w], dtype=torch.float32, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        src_h = torch.clamp_min(tile_h.index.to(torch.float32) * scale_h, 0.0)
        h0 = src_h.to(torch.int32)
        h1 = torch.clamp_max(h0 + 1, in_h - 1)
        src_w = torch.clamp_min(tile_w.index.to(torch.float32) * scale_w, 0.0)
        w0 = src_w.to(torch.int32)
        w1 = torch.clamp_max(w0 + 1, in_w - 1)
        lerp_w = torch.clamp_max(
            torch.clamp_min(src_w - w0.to(torch.float32), 0.0), 1.0
        )
        lerp_h = torch.clamp_max(
            torch.clamp_min(src_h - h0.to(torch.float32), 0.0), 1.0
        )
        # _unsafe_index(x, [None, None, rows[:, None], cols]): four gathers.
        lower_left = x[tile_p, h1[:, None], w0[None, :]]
        lower_right = x[tile_p, h1[:, None], w1[None, :]]
        lower = lower_left + (lower_right - lower_left) * lerp_w[None, None, :]
        upper_left = x[tile_p, h0[:, None], w0[None, :]]
        upper_right = x[tile_p, h0[:, None], w1[None, :]]
        upper = upper_left + (upper_right - upper_left) * lerp_w[None, None, :]
        out[tile_p, tile_h, tile_w] = upper + (lower - upper) * lerp_h[None, :, None]
    return out


@helion.kernel(static_shapes=True)
def cat_skip_padded_upsample(skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # skip: [N, C0, H*W] bf16; up: [N, C1, H, W - pad] fp32. Writes
    # cat([skip, constant_pad_nd(up.to(bf16), [0, pad, 0, 0])], 1) as
    # [N, C0 + C1, H*W].
    batch, skip_channels, plane = skip.shape
    up_channels = up.size(1)
    height = up.size(2)
    up_width = up.size(3)
    width = plane // height
    channels = skip_channels + up_channels
    out = torch.empty([batch, channels, plane], dtype=skip.dtype, device=skip.device)
    for tile_n, tile_c, tile_s in hl.tile([batch, channels, plane]):
        # The column and the row of a plane position, Inductor's x0 and x1:
        # the upsample's rows are one column shorter than the output's, so
        # its element is not at the output's plane position.
        col = tile_s.index % width
        row = tile_s.index // width
        from_skip = (tile_c.index < skip_channels)[None, :, None]
        from_up = (tile_c.index >= skip_channels)[None, :, None] & (col < up_width)[
            None, None, :
        ]
        skip_value = hl.load(skip, [tile_n, tile_c, tile_s], extra_mask=from_skip).to(
            torch.float32
        )
        up_value = hl.load(
            up,
            [
                tile_n,
                (tile_c.index - skip_channels)[:, None],
                row[None, :],
                col[None, :],
            ],
            extra_mask=from_up,
        )
        padded = torch.where(from_up, up_value, 0.0)
        out[tile_n, tile_c, tile_s] = torch.where(from_skip, skip_value, padded).to(
            out.dtype
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    mean, x, var, weight, bias, skip = inputs[:6]
    batch, channels, height, width = x.shape
    skip_channels, out_h, out_w = skip.shape[1:]
    # Before the first launch: a skip connection whose rows do not follow
    # one another in memory (one cropped along its width, say) has no such view.
    skip_planes = skip.view(batch, skip_channels, out_h * out_w)
    act = batchnorm_relu(
        x.view(batch, channels, height * width), mean, var, weight, bias, 1e-05
    )
    # iota(320) and iota(478) scaled by the graph's align_corners factors,
    # (H - 1) / (320 - 1) and (W - 1) / (478 - 1).
    up = upsample_bilinear(
        act.view(batch * channels, height, width),
        320,
        478,
        0.49843260188087773,
        0.4989517819706499,
    )
    out = cat_skip_padded_upsample(skip_planes, up.view(batch, channels, 320, 478))
    return (out.view(batch, skip_channels + channels, out_h, out_w),)
