"""better-benchmark repro sum_sum_cd8694c00507 (timm mobilevit_s training).

SiLU backward feeding the BatchNorm training backward on channels-last bf16
activations (mobilevit_s and tf_efficientnet_b0 training). The BatchNorm
output is recomputed from its saved statistics,
``y = (x - mean) * invstd * weight + bias``; the SiLU backward is
``grad = grad_out * sigmoid(y) * (y * (1 - sigmoid(y)) + 1)``; the BatchNorm
backward takes the per-channel sums ``sum(grad)`` and
``sum(grad * (x - mean))`` over (N, H, W) and returns ``sum(grad)``, the
weight gradient ``sum(grad * (x - mean)) * invstd`` and the input gradient
``(grad - (x - mean) * (sum_dot * c) * invstd**2 - sum_grad * c) * invstd
* weight`` in bf16; ``c`` is the graph literal 4.76837158203125e-07 (1 /
2097152) at every shape point, not 1 / (N*H*W). Inductor emits five kernels:

- ``triton_poi_fused_add_convert_element_type_mul_sub_unsqueeze_0``: writes
  ``y`` to an fp32 buffer shaped like the activations.
- ``triton_red_fused_add_convert_element_type_mul_sigmoid_squeeze_sub_sum_unsqueeze_1``:
  split reduction; each block of ``rows_per_split`` rows of the
  ``[N*H*W, C]`` view writes its per-channel partial
  ``sum(grad * (x - mean))`` and ``sum(grad)`` to two fp32 ``[splits, C]``
  workspaces.
- ``triton_red_fused_add_convert_element_type_mul_sigmoid_squeeze_sub_sum_unsqueeze_2``:
  sums the ``grad * (x - mean)`` partials per channel and stores the sum and
  the weight gradient (the sum times ``invstd``).
- ``triton_red_fused_add_convert_element_type_mul_sigmoid_sub_sum_3``: sums
  the ``grad`` partials per channel.
- ``triton_poi_fused_add_convert_element_type_mul_sigmoid_squeeze_sub_unsqueeze_4``:
  recomputes ``grad`` and ``x - mean`` and writes the input gradient.

Like Inductor (no ``emulate_precision_casts``), the graph's bf16 round trips
of ``y`` and of ``grad`` stay in fp32.

The split is Inductor's at shape point 65ca7fc0: 591 blocks of 888 rows, the
last one partial (524288 rows). Elsewhere the port keeps 888-row blocks, so
the split kernel loops over a block's rows with an explicit ``hl.tile`` and
masked loads, as Inductor's looped split reduction does. The kernels read
``[N*H*W, C]`` views of channels-last activations, which every shape point
has; for another layout the view fails before the first launch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split at shape point 65ca7fc0: blocks of 888 rows (591 of them).
ROWS_PER_SPLIT = 888


@helion.kernel(static_shapes=True)
def bn_affine(
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # bn_input: [N*H*W, C] view of the channels-last activations; the rest: [C].
    rows, channels = bn_input.shape
    out = torch.empty([rows, channels], dtype=torch.float32, device=bn_input.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        centered = bn_input[tile_r, tile_c].to(torch.float32) - mean[tile_c]
        out[tile_r, tile_c] = centered * invstd[tile_c] * weight[tile_c] + bias[tile_c]
    return out


@helion.kernel(static_shapes=True)
def silu_bn_bwd_partial_sums(
    grad_out: torch.Tensor,
    bn_out: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    rows_per_split: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad_out, bn_input: [N*H*W, C] views of the channels-last activations;
    # bn_out: [N*H*W, C] fp32; mean: [C].
    rows, channels = grad_out.shape
    splits = (rows + rows_per_split - 1) // rows_per_split
    dot_partial = torch.empty(
        [splits, channels], dtype=torch.float32, device=grad_out.device
    )
    grad_partial = torch.empty(
        [splits, channels], dtype=torch.float32, device=grad_out.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        mean_c = mean[tile_c]
        dot = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        total = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            g = hl.load(grad_out, [row, tile_c], extra_mask=mask).to(torch.float32)
            y = hl.load(bn_out, [row, tile_c], extra_mask=mask)
            sig = torch.sigmoid(y)
            grad = g * sig * (y * (1.0 - sig) + 1.0)
            x = hl.load(bn_input, [row, tile_c], extra_mask=mask).to(torch.float32)
            centered = x - mean_c[None, None, :]
            dot = dot + torch.sum(torch.where(mask, grad * centered, 0.0), dim=1)
            total = total + torch.sum(torch.where(mask, grad, 0.0), dim=1)
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
def silu_bn_bwd_grad_input(
    grad_out: torch.Tensor,
    bn_out: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    dot_sum: torch.Tensor,
    invstd: torch.Tensor,
    grad_sum: torch.Tensor,
    weight: torch.Tensor,
    inv_count: hl.constexpr,
) -> torch.Tensor:
    # grad_out, bn_input: [N*H*W, C] views of the channels-last activations;
    # bn_out: [N*H*W, C] fp32; the rest: [C].
    rows, channels = grad_out.shape
    out = torch.empty([rows, channels], dtype=grad_out.dtype, device=grad_out.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y = bn_out[tile_r, tile_c]
        sig = torch.sigmoid(y)
        grad = (
            grad_out[tile_r, tile_c].to(torch.float32) * sig * (y * (1.0 - sig) + 1.0)
        )
        centered = bn_input[tile_r, tile_c].to(torch.float32) - mean[tile_c]
        scale = invstd[tile_c]
        projection = dot_sum[tile_c] * inv_count * (scale * scale)
        result = grad - centered * projection
        result = result - grad_sum[tile_c] * inv_count
        result = result * (scale * weight[tile_c])
        out[tile_r, tile_c] = result.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_out, bn_input, mean, invstd, weight, bias = inputs[:6]
    n, c, h, w = bn_input.shape
    rows = n * h * w

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 3, 1).view(rows, c)

    mean_c = mean.view(c)
    invstd_c = invstd.view(c)
    bn_out = bn_affine(flat(bn_input), mean_c, invstd_c, weight, bias)
    dot_partial, grad_partial = silu_bn_bwd_partial_sums(
        flat(grad_out), bn_out, flat(bn_input), mean_c, ROWS_PER_SPLIT
    )
    dot_sum, weight_grad = bn_bwd_dot_sum_weight_grad(dot_partial, invstd_c)
    grad_sum = bn_bwd_grad_sum(grad_partial)
    grad_input = silu_bn_bwd_grad_input(
        flat(grad_out),
        bn_out,
        flat(bn_input),
        mean_c,
        dot_sum,
        invstd_c,
        grad_sum,
        weight,
        4.76837158203125e-07,
    )
    grad_input = grad_input.view(n, h, w, c).permute(0, 3, 1, 2)
    return grad_sum, weight_grad, grad_input
