"""better-benchmark repro sum_sum_2a737da66d69 (torchbench resnet152 training et al.).

A ResNet block's residual join backward feeding the BatchNorm training
backward on channels-last bf16 activations (resnet18/50/152, resnext50,
LearningToPaint and phlippe_resnet training): the two incoming gradients are
added and masked by the ReLU, ``grad = where(relu_out <= 0, fill, grad_a +
grad_b)`` (``fill`` a 0-d bf16 tensor), returned in bf16; the BatchNorm
backward takes the per-channel sums ``sum(grad)`` and
``sum(grad * (x - mean))`` over (N, H, W) and returns ``sum(grad)``, the
weight gradient ``sum(grad * (x - mean)) * invstd`` and the input gradient
``(grad - (x - mean) * (sum_dot / count) * invstd**2 - sum_grad / count)
* invstd * weight`` in bf16 (``1 / count`` is the graph literal
1.0172526041666666e-05, 1 / 98304 from the captured 96x32x32 activation, at
every shape point). Inductor emits five kernels:

- ``triton_poi_fused_add_le_where_0``: writes ``grad`` in bf16. The three
  activations and the output share one dense layout, so Inductor indexes them
  as one flat dimension (three loads, the 0-d fill and one store per
  element); the port tiles the same flat views.
- ``triton_red_fused_convert_element_type_mul_sub_sum_1``: split reduction;
  a looped reduction over H*W per (n, c) writes each sample's per-channel
  partial ``sum(grad * (x - mean))`` and ``sum(grad)`` to two fp32 ``[N, C]``
  workspaces.
- ``triton_red_fused_convert_element_type_mul_sub_sum_2``: sums the
  ``grad * (x - mean)`` partials per channel and stores the sum and the weight
  gradient (the sum times ``invstd``).
- ``triton_red_fused_convert_element_type_sum_3``: sums the ``grad`` partials
  per channel.
- ``triton_poi_fused_convert_element_type_mul_sub_unsqueeze_4``: reloads
  ``grad``, recomputes ``x - mean`` and writes the input gradient.

The split is Inductor's at shape point ff6e3deb: one block per sample (128
blocks of 14*14 = 196 rows of the ``[N*H*W, C]`` view), so the split kernel
reads ``[N, H*W, C]`` views and sums a whole dim of them. The kernels read
views of channels-last activations, which 12 of the 21 shape points have; at
the 9 points with an NCHW activation the view fails before the first launch.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def relu_bwd_add(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    relu_out: torch.Tensor,
    fill: torch.Tensor,
) -> torch.Tensor:
    # grad_a, grad_b, relu_out: [N*H*W*C] bf16, flat views of the
    # channels-last activations; fill: 0-d bf16.
    numel = relu_out.size(0)
    out = torch.empty([numel], dtype=grad_a.dtype, device=grad_a.device)
    for tile in hl.tile(numel):
        total = grad_a[tile].to(torch.float32) + grad_b[tile].to(torch.float32)
        grad = torch.where(
            relu_out[tile].to(torch.float32) <= 0.0,
            fill[()].to(torch.float32),
            total,
        )
        out[tile] = grad.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def bn_bwd_partial_sums(
    grad_rows: torch.Tensor, x_rows: torch.Tensor, mean: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad_rows, x_rows: [N, H*W, C] bf16, views of the channels-last
    # gradient and BatchNorm input; mean: [C] fp32.
    n, hw, channels = grad_rows.shape
    dot_partial = torch.empty(
        [n, channels], dtype=torch.float32, device=grad_rows.device
    )
    grad_partial = torch.empty(
        [n, channels], dtype=torch.float32, device=grad_rows.device
    )
    for tile_n, tile_c in hl.tile([n, channels]):
        grad = grad_rows[tile_n, :, tile_c].to(torch.float32)
        centered = x_rows[tile_n, :, tile_c].to(torch.float32) - mean[tile_c]
        dot_partial[tile_n, tile_c] = torch.sum(grad * centered, dim=1)
        grad_partial[tile_n, tile_c] = torch.sum(grad, dim=1)
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
def bn_bwd_grad_input(
    grad: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    dot_sum: torch.Tensor,
    invstd: torch.Tensor,
    grad_sum: torch.Tensor,
    weight: torch.Tensor,
    inv_count: hl.constexpr,
) -> torch.Tensor:
    # grad, bn_input: [N*H*W, C] bf16, views of the channels-last gradient
    # and BatchNorm input; the rest: [C] fp32.
    rows, channels = grad.shape
    out = torch.empty([rows, channels], dtype=grad.dtype, device=grad.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        centered = bn_input[tile_r, tile_c].to(torch.float32) - mean[tile_c]
        scale = invstd[tile_c]
        projection = dot_sum[tile_c] * inv_count * (scale * scale)
        y = grad[tile_r, tile_c].to(torch.float32) - centered * projection
        y = y - grad_sum[tile_c] * inv_count
        y = y * (scale * weight[tile_c])
        out[tile_r, tile_c] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_a, grad_b, relu_out, fill, bn_input, mean, invstd, weight = inputs[:8]
    n, c, h, w = relu_out.shape
    rows = n * h * w

    def nhwc(t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 3, 1)

    # Every view is taken before the first launch, so a layout the port does
    # not handle fails here.
    flat_inputs = [nhwc(t).view(-1) for t in (grad_a, grad_b, relu_out)]
    x_nhwc = nhwc(bn_input)
    x_blocks = x_nhwc.view(n, h * w, c)
    x_rows = x_nhwc.view(rows, c)
    mean_c = mean.view(c)

    grad = relu_bwd_add(*flat_inputs, fill)
    dot_partial, grad_partial = bn_bwd_partial_sums(
        grad.view(n, h * w, c), x_blocks, mean_c
    )
    dot_sum, weight_grad = bn_bwd_dot_sum_weight_grad(dot_partial, invstd)
    grad_sum = bn_bwd_grad_sum(grad_partial)
    grad_input = bn_bwd_grad_input(
        grad.view(rows, c),
        x_rows,
        mean_c,
        dot_sum,
        invstd,
        grad_sum,
        weight,
        1.0172526041666666e-05,
    )
    return (
        grad.view(n, h, w, c).permute(0, 3, 1, 2),
        grad_sum,
        weight_grad,
        grad_input.view(n, h, w, c).permute(0, 3, 1, 2),
    )
