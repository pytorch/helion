"""better-benchmark repro sum_cf864ca94168 (torchbench/train/vgg16).

The max-pool backward followed by the ReLU backward and the bias gradient of
the convolution before it, on channels-last bf16 activations: the pooled
gradient ``[N, C, H, W]`` is scattered into a zero fp32 ``[N*C, in_h*in_w]``
buffer at the input position each pooling window took its maximum from
(``_low_memory_max_pool_offsets_to_indices`` of the int8 window offsets,
then ``scatter_add``), ``where(mask, fill, scattered)`` (``fill`` a 0-d bf16
tensor) is stored in bf16 channels-last, and its per-channel sum over
(N, H, W) is returned in fp32. Inductor emits five kernels:

- ``triton_poi_fused__low_memory_max_pool_offsets_to_indices__unsafe_view_clone_convert_element_type_full_scatter_add_0``:
  zero-fills the fp32 scatter buffer, indexed as one flat dimension.
- ``triton_poi_fused__low_memory_max_pool_offsets_to_indices__unsafe_view_clone_convert_element_type_full_scatter_add_1``:
  for every pooled element, loads its int8 offset and its bf16 gradient
  straight from the channels-last inputs (the graph's contiguous clones are
  fused away), wraps a negative offset by the window size, splits it into
  the window row and column, and atomically adds the gradient (as fp32) to
  ``buf[n*C + c, in_row * in_w + in_col]`` with
  ``in_row = h * stride + window_row * dilation - padding`` (padding 0 and
  dilation 1 are graph literals). Its index math is int32.
- ``triton_poi_fused_convert_element_type_view_where_2``: writes the bf16
  channels-last ``where`` output in its memory order, loading the bool mask
  in that order, the fill once, and the fp32 scatter buffer transposed
  (``[N, C, in_h*in_w]`` read at ``[n, s, c]``). As in Inductor, the graph's
  bf16 cast of the scattered value happens at the store.
- ``triton_red_fused_sum_3``: the first level of the split per-channel sum
  over the ``[N*in_h*in_w, C]`` view of that output: blocks of
  ``ROWS_PER_SPLIT`` (2048) consecutive rows write their per-channel partial
  to an fp32 ``[splits, C]`` workspace.
- ``triton_red_fused_convert_element_type_sum_4``: sums the partials per
  channel into the fp32 ``[C]`` output. As in Inductor (no
  ``emulate_precision_casts``), the graph's bf16 round trip of that sum stays
  in fp32.

The split is Inductor's at shape point b721890d: 1568 blocks of 2048 rows
(3211264 rows). The blocks divide the rows evenly there, so the split kernel
sums a whole dim of a ``[splits, 2048, C]`` view; ``forward()`` rejects shape
points whose rows are not a multiple of 2048 before its first launch. The
scatter and ``where`` kernels tile their own dims where Inductor flattens
them and recovers the row and column with divisions and remainders.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split at shape point b721890d: blocks of 2048 rows (1568 of them).
ROWS_PER_SPLIT = 2048


@helion.kernel(static_shapes=True)
def zero_fill(out: torch.Tensor) -> torch.Tensor:
    # out: the flat view of the dense fp32 scatter buffer.
    n = out.size(0)
    for tile_n in hl.tile(n):
        out[tile_n] = hl.zeros([tile_n], dtype=out.dtype)
    return out


@helion.kernel(static_shapes=True)
def max_unpool_scatter_add(
    offsets: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    kernel_h: hl.constexpr,
    kernel_w: hl.constexpr,
    in_w: hl.constexpr,
    stride_h: hl.constexpr,
    stride_w: hl.constexpr,
    pad_h: hl.constexpr,
    pad_w: hl.constexpr,
    dil_h: hl.constexpr,
    dil_w: hl.constexpr,
) -> torch.Tensor:
    # offsets (int8), grad (bf16): [N, H, W, C] views of the channels-last
    # inputs; out: [N, C, in_h * in_w] fp32, zero-filled.
    batch, height, width, channels = grad.shape
    window = kernel_h * kernel_w
    for tile_n, tile_h, tile_w, tile_c in hl.tile([batch, height, width, channels]):
        offset = offsets[tile_n, tile_h, tile_w, tile_c].to(torch.int32)
        offset = torch.where(offset < 0, offset + window, offset)
        in_row = (
            (tile_h.index * stride_h)[None, :, None, None]
            + (offset // kernel_w) * dil_h
            - pad_h
        )
        in_col = (
            (tile_w.index * stride_w)[None, None, :, None]
            + (offset % kernel_w) * dil_w
            - pad_w
        )
        # out[n, c, in_row * in_w + in_col] += grad[n, h, w, c]
        hl.atomic_add(
            out,
            [
                tile_n.index[:, None, None, None],
                tile_c.index[None, None, None, :],
                in_row * in_w + in_col,
            ],
            grad[tile_n, tile_h, tile_w, tile_c].to(torch.float32),
        )
    return out


@helion.kernel(static_shapes=True)
def where_fill(
    mask: torch.Tensor, fill: torch.Tensor, scattered: torch.Tensor
) -> torch.Tensor:
    # mask: [N, in_h*in_w, C] bool, a view of the channels-last input; fill:
    # 0-d bf16; scattered: [N, in_h*in_w, C] fp32, the permuted view of the
    # [N, C, in_h*in_w] scatter buffer.
    batch, plane, channels = mask.shape
    out = torch.empty([batch, plane, channels], dtype=fill.dtype, device=mask.device)
    for tile_n, tile_s, tile_c in hl.tile([batch, plane, channels]):
        out[tile_n, tile_s, tile_c] = torch.where(
            mask[tile_n, tile_s, tile_c],
            fill[()].to(torch.float32),
            scattered[tile_n, tile_s, tile_c],
        ).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def channel_partial_sums(x_blocks: torch.Tensor) -> torch.Tensor:
    # x_blocks: [splits, rows_per_split, C] bf16, a view of the channels-last
    # where output; returns each block's per-channel sum as fp32 [splits, C].
    splits, rows_per_split, channels = x_blocks.shape
    partials = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_blocks.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        partials[tile_s, tile_c] = torch.sum(
            x_blocks[tile_s, :, tile_c].to(torch.float32), dim=1
        )
    return partials


@helion.kernel(static_shapes=True)
def sum_partials(partials: torch.Tensor) -> torch.Tensor:
    # The graph rounds this sum to bf16 and back; Inductor keeps it in fp32.
    splits, channels = partials.shape
    out = torch.empty([channels], dtype=torch.float32, device=partials.device)
    for tile_c in hl.tile(channels):
        out[tile_c] = torch.sum(partials[:, tile_c], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, offsets, mask, fill = inputs[:4]
    scatter_shape = inputs[4]  # _shape_param_0 = [N*C, in_h*in_w]
    kernel_h, kernel_w = inputs[6]  # _shape_param_2 = kernel_size
    in_h, in_w = inputs[7]  # _shape_param_3 = input_size
    stride_h, stride_w = inputs[8]  # _shape_param_4 = stride
    n, c, h, w = grad.shape
    plane = in_h * in_w
    rows = n * plane
    if rows % ROWS_PER_SPLIT != 0:
        raise ValueError(
            f"this port splits the channel sum into {ROWS_PER_SPLIT}-row blocks; "
            f"{rows} rows are not a multiple"
        )

    scattered = torch.empty(scatter_shape, dtype=torch.float32, device=grad.device)
    zero_fill(scattered.view(-1))
    max_unpool_scatter_add(
        offsets.permute(0, 2, 3, 1),
        grad.permute(0, 2, 3, 1),
        scattered.view(n, c, plane),
        kernel_h,
        kernel_w,
        in_w,
        stride_h,
        stride_w,
        0,
        0,
        1,
        1,
    )
    where = where_fill(
        mask.permute(0, 2, 3, 1).view(n, plane, c),
        fill,
        scattered.view(n, c, plane).permute(0, 2, 1),
    )
    partials = channel_partial_sums(
        where.view(rows // ROWS_PER_SPLIT, ROWS_PER_SPLIT, c)
    )
    channel_sum = sum_partials(partials)
    return where.view(n, in_h, in_w, c).permute(0, 3, 1, 2), channel_sum
