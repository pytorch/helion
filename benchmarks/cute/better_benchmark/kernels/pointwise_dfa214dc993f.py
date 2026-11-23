"""better-benchmark repro pointwise_dfa214dc993f (torchbench squeezenet1_1).

The end of a SqueezeNet Fire module and the pooling after it: the ReLUs of the
two expand branches (contiguous bf16 ``[N, C_i, H, W]`` activations), their
concatenation along the channels, and a max pooling without padding (3x3
window, stride 2, dilation 1, ceil mode; the pooling's offsets output is
dead). The pooling reads each element of the concatenation from several
windows, so Inductor materializes it and emits two pointwise kernels:

- ``triton_poi_fused_cat_relu_0``: a pointwise kernel over the
  ``[N, C0 + C1, H * W]`` concatenation, indexed as one flat dimension with
  the channel and the position in the plane recovered by divisions and
  remainders. Each branch is one load masked to the branch's channel range,
  so an element reads one activation; the NaN-propagating ReLUs are computed
  in fp32, merged with a ``where`` and stored in bf16 to a temporary. The
  port tiles the same three dimensions (the activations are
  ``[N, C_i, H * W]`` views) with the same masked loads.
- ``triton_poi_fused__low_memory_max_pool_with_offsets_cat_relu_1``: each
  element of the contiguous ``[N, C0 + C1, OH, OW]`` output loads its
  ``kernel_h * kernel_w`` window of the bf16 temporary (nine unmasked loads:
  at these shape points the ceil-mode windows end exactly at the image's
  edge) and folds the fp32 values with a NaN-propagating maximum in
  row-major window order before the bf16 store. Inductor indexes the output
  as one flat dimension and recovers ``(n * C + c, oh, ow)`` from it; the
  port tiles those three dimensions of the same storage (``[N*C, H, W]`` and
  ``[N*C, OH, OW]`` views) and gathers the window with the same affine index
  math, ``(oh * stride_h + i, ow * stride_w + j)``.

The temporary between the two kernels has Inductor's layout: comprehensive
padding rounds its per-channel plane stride up to 128 bytes (``H * W = 3025``
to 3072 bf16 elements at the 55x55 shape point; the 27x27 planes are below the
padding threshold and stay contiguous), so every plane the first kernel stores
starts aligned. The port allocates that storage on the host and passes the
first kernel the ``[N, C0 + C1, H * W]`` alias of it.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's comprehensive padding (``padding_alignment_bytes`` and
# ``padding_stride_threshold``): a temporary's stride above 1024 elements is
# rounded up to a multiple of 128 bytes.
PADDING_ALIGNMENT_BYTES = 128
PADDING_STRIDE_THRESHOLD = 1024


@helion.kernel(static_shapes=True)
def relu_cat(x0: torch.Tensor, x1: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # x0, x1: [N, C_i, H*W] bf16 views of the contiguous activations; out:
    # [N, C0 + C1, H*W] bf16, the temporary (padded plane stride). Writes
    # cat([relu(x0), relu(x1)], 1).
    batch, channels0, plane = x0.shape
    channels = channels0 + x1.size(1)
    for tile_n, tile_c, tile_s in hl.tile([batch, channels, plane]):
        c = tile_c.index
        from_x0 = (c < channels0)[None, :, None]
        from_x1 = (c >= channels0)[None, :, None]
        y0 = torch.relu(
            hl.load(x0, [tile_n, tile_c, tile_s], extra_mask=from_x0).to(torch.float32)
        )
        y1 = torch.relu(
            hl.load(x1, [tile_n, c - channels0, tile_s], extra_mask=from_x1).to(
                torch.float32
            )
        )
        out[tile_n, tile_c, tile_s] = torch.where(from_x0, y0, y1).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def max_pool(
    x: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
) -> torch.Tensor:
    # x: [N*C, H, W] bf16, a view of the temporary (padded plane stride).
    planes, height, width = x.shape
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    out = torch.empty([planes, out_h, out_w], dtype=x.dtype, device=x.device)
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        row = (tile_h.index * stride_h)[:, None]
        col = (tile_w.index * stride_w)[None, :]
        best = x[tile_p, row, col].to(torch.float32)
        for offset in hl.static_range(1, kernel_h * kernel_w):
            i = offset // kernel_w
            j = offset % kernel_w
            value = x[tile_p, row + i, col + j].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_p, tile_h, tile_w] = best.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    kernel_h, kernel_w = inputs[2]  # _shape_param_0 = kernel_size
    stride_h, stride_w = inputs[3]  # _shape_param_1 = stride
    n, c0, h, w = arg0_1.shape
    c1 = arg1_1.size(1)
    channels = c0 + c1
    plane = h * w
    # The graph's padding is the literal [0, 0], its dilation [1, 1] and its
    # ceil mode True: a last window that overhangs the image would need
    # masked loads, which Inductor's kernel does not have at these shapes.
    if (h - kernel_h) % stride_h != 0 or (w - kernel_w) % stride_w != 0:
        raise ValueError(
            f"ceil-mode windows overhang the {h}x{w} image "
            f"(kernel {kernel_h}x{kernel_w}, stride {stride_h}x{stride_w})"
        )
    # The temporary's plane stride, as Inductor pads it.
    align = PADDING_ALIGNMENT_BYTES // arg0_1.dtype.itemsize
    plane_stride = plane
    if plane_stride > PADDING_STRIDE_THRESHOLD:
        plane_stride = (plane_stride + align - 1) // align * align
    cat = torch.empty(
        [n, channels, plane_stride], dtype=arg0_1.dtype, device=arg0_1.device
    )[:, :, :plane]
    relu_cat(arg0_1.view(n, c0, plane), arg1_1.view(n, c1, plane), cat)
    out = max_pool(cat.view(n * channels, h, w), kernel_h, kernel_w, stride_h, stride_w)
    return (out.view(n, channels, out.size(1), out.size(2)),)
