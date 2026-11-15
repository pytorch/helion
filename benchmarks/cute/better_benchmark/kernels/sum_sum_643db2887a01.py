"""better-benchmark repro sum_sum_643db2887a01 (timm inception_v3 training).

ReLU backward feeding the BatchNorm training backward on channels-last bf16
activations: ``grad = where(relu_out <= 0, fill, grad_out)`` (``fill`` a 0-d
bf16 tensor), the per-channel sums ``sum(grad)`` and
``sum(grad * (x - mean))`` over (N, H, W), the weight gradient
``sum(grad * (x - mean)) * invstd`` and the input gradient
``(grad - (x - mean) * (sum_dot / count) * invstd**2 - sum_grad / count)
* invstd * weight`` in bf16 (``1 / count`` is the graph literal
3.5189856312778704e-07). Inductor emits four kernels:

- ``triton_red_fused_convert_element_type_le_mul_sub_sum_where_0``: split
  reduction; each block of ``rows_per_split`` rows of the ``[N*H*W, C]``
  view writes its per-channel partial ``sum(grad * (x - mean))`` and
  ``sum(grad)`` to two fp32 ``[splits, C]`` workspaces.
- ``triton_red_fused_convert_element_type_le_mul_sub_sum_where_1``: sums the
  ``grad * (x - mean)`` partials per channel and stores the sum and the weight
  gradient (the sum times ``invstd``).
- ``triton_red_fused_convert_element_type_le_sum_where_2``: sums the
  ``grad`` partials per channel.
- ``triton_poi_fused_convert_element_type_le_mul_sub_unsqueeze_where_3``:
  recomputes ``grad`` and ``x - mean`` and writes the input gradient.

The split is Inductor's at the default shape point 2cd54028: 1192 blocks of
2384 rows. Elsewhere the port keeps 2384-row blocks, the last one partial,
so the split kernel loops over a block's rows with an explicit ``hl.tile``
and masked loads, as Inductor's looped split reductions do. The kernels read
``[N*H*W, C]`` views of channels-last activations, which 57 of the 74 shape
points have; at the 17 NCHW points the view fails before the first launch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split at shape point 2cd54028: blocks of 2384 rows (1192 of them).
ROWS_PER_SPLIT = 2384


@helion.kernel(static_shapes=True)
def relu_bn_bwd_partial_sums(
    relu_out: torch.Tensor,
    fill: torch.Tensor,
    grad_out: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    rows_per_split: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # relu_out, grad_out, bn_input: [N*H*W, C] views of the channels-last
    # activations; fill: 0-d; mean: [C].
    rows, channels = relu_out.shape
    splits = (rows + rows_per_split - 1) // rows_per_split
    dot_partial = torch.empty(
        [splits, channels], dtype=torch.float32, device=relu_out.device
    )
    grad_partial = torch.empty(
        [splits, channels], dtype=torch.float32, device=relu_out.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        fill_value = fill[()].to(torch.float32)
        mean_c = mean[tile_c]
        dot = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        total = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            relu = hl.load(relu_out, [row, tile_c], extra_mask=mask)
            grad = torch.where(
                relu.to(torch.float32) <= 0.0,
                fill_value,
                hl.load(grad_out, [row, tile_c], extra_mask=mask).to(torch.float32),
            )
            grad = torch.where(mask, grad, 0.0)
            x = hl.load(bn_input, [row, tile_c], extra_mask=mask).to(torch.float32)
            centered = x - mean_c[None, None, :]
            dot = dot + torch.sum(grad * centered, dim=1)
            total = total + torch.sum(grad, dim=1)
        dot_partial[tile_s, tile_c] = dot
        grad_partial[tile_s, tile_c] = total
    return dot_partial, grad_partial


@helion.kernel(static_shapes=True)
def bn_bwd_dot_sum_weight_grad(
    dot_partial: torch.Tensor, invstd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    splits, channels = dot_partial.shape
    dot_sum = torch.empty([channels], dtype=torch.float32, device=dot_partial.device)
    weight_grad = torch.empty(
        [channels], dtype=torch.float32, device=dot_partial.device
    )
    for tile_c in hl.tile(channels):
        total = torch.sum(dot_partial[:, tile_c], dim=0)
        dot_sum[tile_c] = total
        weight_grad[tile_c] = total * invstd[tile_c]
    return dot_sum, weight_grad


@helion.kernel(static_shapes=True)
def bn_bwd_grad_sum(grad_partial: torch.Tensor) -> torch.Tensor:
    splits, channels = grad_partial.shape
    grad_sum = torch.empty([channels], dtype=torch.float32, device=grad_partial.device)
    for tile_c in hl.tile(channels):
        grad_sum[tile_c] = torch.sum(grad_partial[:, tile_c], dim=0)
    return grad_sum


@helion.kernel(static_shapes=True)
def relu_bn_bwd_grad_input(
    relu_out: torch.Tensor,
    fill: torch.Tensor,
    grad_out: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    dot_sum: torch.Tensor,
    invstd: torch.Tensor,
    grad_sum: torch.Tensor,
    weight: torch.Tensor,
    inv_count: hl.constexpr,
) -> torch.Tensor:
    # relu_out, grad_out, bn_input: [N*H*W, C] views of the channels-last
    # activations; fill: 0-d; the rest: [C].
    rows, channels = relu_out.shape
    out = torch.empty([rows, channels], dtype=grad_out.dtype, device=grad_out.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        grad = torch.where(
            relu_out[tile_r, tile_c].to(torch.float32) <= 0.0,
            fill[()].to(torch.float32),
            grad_out[tile_r, tile_c].to(torch.float32),
        )
        centered = bn_input[tile_r, tile_c].to(torch.float32) - mean[tile_c]
        scale = invstd[tile_c]
        projection = dot_sum[tile_c] * inv_count * (scale * scale)
        y = grad - centered * projection
        y = y - grad_sum[tile_c] * inv_count
        y = y * (scale * weight[tile_c])
        out[tile_r, tile_c] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    relu_out, fill, grad_out, bn_input, mean, invstd, weight = inputs[:7]
    n, c, h, w = relu_out.shape
    rows = n * h * w

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 3, 1).view(rows, c)

    mean_c = mean.view(c)
    dot_partial, grad_partial = relu_bn_bwd_partial_sums(
        flat(relu_out), fill, flat(grad_out), flat(bn_input), mean_c, ROWS_PER_SPLIT
    )
    dot_sum, weight_grad = bn_bwd_dot_sum_weight_grad(dot_partial, invstd)
    grad_sum = bn_bwd_grad_sum(grad_partial)
    grad_input = relu_bn_bwd_grad_input(
        flat(relu_out),
        fill,
        flat(grad_out),
        flat(bn_input),
        mean_c,
        dot_sum,
        invstd,
        grad_sum,
        weight,
        3.5189856312778704e-07,
    )
    grad_input = grad_input.view(n, h, w, c).permute(0, 3, 1, 2)
    return grad_sum, weight_grad, grad_input
