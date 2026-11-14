"""better-benchmark repro var_mean_var_mean_5cc92f5d49c1 (torchbench resnet50 training).

Two BatchNorm training forwards (eps 1e-5, momentum 0.1) on NCHW bf16
activations, each updating its running mean and running variance inputs in
place, whose affine outputs are added and passed through ReLU (a ResNet
bottleneck's residual join with a BatchNorm on the shortcut). Inductor emits
five kernel launches of three kernels:

- ``triton_red_fused_convert_element_type_var_mean_0`` (once per input): the
  first level of a split ``var_mean`` over (N, H, W). Each channel's
  ``N*H*W`` elements, flattened in (n, h, w) order, are cut into ``splits``
  (4) blocks of ``rows_per_split`` consecutive elements, and a looped Welford
  reduction writes each block's per-channel mean, m2 and weight to three fp32
  ``[splits, C]`` workspaces. A block spans whole images at this shape point
  (8 x 56 x 56), which is not one tensor dim of the NCHW input, so the port
  loops over the flattened block index with Inductor's index math
  (``n = r // (H*W)``, ``hw = r % (H*W)``), combining each chunk's
  statistics into running Welford accumulators.
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_1``
  (once per input): a persistent combine of the partials per channel into the
  mean and ``rsqrt(m2 / (N*H*W) + 1e-5)``, updating the running statistics in
  place (the graph's Bessel factor is the literal 1.0000398612827361, from the
  captured 128x14x14 activation, at every shape point).
- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_2``:
  ``relu(((x0 - mean0) * rstd0 * weight0 + bias0) + ((x1 - mean1) * rstd1 *
  weight1 + bias1))`` stored in bf16, over the flat contiguous activations
  with the channel ``(i // (H*W)) % C`` gathering the per-channel operands.
  Like Inductor (no ``emulate_precision_casts``), the graph's bf16 rounding of
  each affine output before the add stays in fp32.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split of each channel's (N, H, W) reduction for this shape: 4
# blocks of 25088 elements (8 images of 56x56).
SPLITS = 4


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor, splits: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [N, H*W, C] bf16, a permuted view of the NCHW input.
    batch, hw, channels = x_rows.shape
    total = batch * hw
    rows_per_split = (total + splits - 1) // splits
    mean_p = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_rows.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        mean = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        m2 = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        weight = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            # r: index into the channel's flattened (n, h, w) elements.
            r = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (r < total) & (tile_r.index[None, :] < rows_per_split)
            values = hl.load(
                x_rows, [r // hw, r % hw, tile_c], extra_mask=valid[:, :, None]
            ).to(torch.float32)
            count = torch.sum(valid.to(torch.float32), dim=1)[:, None]
            chunk_mean = torch.where(count > 0.0, torch.sum(values, dim=1) / count, 0.0)
            centered = torch.where(
                valid[:, :, None], values - chunk_mean[:, None, :], 0.0
            )
            chunk_m2 = torch.sum(centered * centered, dim=1)
            new_weight = weight + count
            ratio = torch.where(new_weight == 0.0, 0.0, count / new_weight)
            delta = chunk_mean - mean
            mean = mean + delta * ratio
            m2 = m2 + chunk_m2 + delta * delta * weight * ratio
            weight = new_weight
        mean_p[tile_s, tile_c] = mean
        m2_p[tile_s, tile_c] = m2
        weight_p[tile_s, tile_c] = weight
    return mean_p, m2_p, weight_p


@helion.kernel(static_shapes=True)
def bn_welford_combine_update(
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: hl.constexpr,
    momentum: hl.constexpr,
    unbias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # mean_p, m2_p, weight_p: [splits, C] fp32 partials; running_*: [C] fp32,
    # updated in place.
    splits, channels = mean_p.shape
    mean_out = torch.empty([channels], dtype=torch.float32, device=mean_p.device)
    rstd_out = torch.empty([channels], dtype=torch.float32, device=mean_p.device)
    for tile_c in hl.tile(channels):
        means = mean_p[:, tile_c]
        m2s = m2_p[:, tile_c]
        weights = weight_p[:, tile_c]
        total = torch.sum(weights, dim=0)
        mean = torch.sum(means * weights, dim=0) / total
        delta = means - mean[None, :]
        m2 = torch.sum(m2s + weights * delta * delta, dim=0)
        # The weights sum to N*H*W exactly: Inductor's true division by that
        # count (a constexpr divisor would lower to a reciprocal multiply).
        var = m2 / total
        rstd_out[tile_c] = torch.rsqrt(var + eps)
        running_var[tile_c] = var * unbias * momentum + running_var[tile_c] * (
            1.0 - momentum
        )
        running_mean[tile_c] = mean * momentum + running_mean[tile_c] * (1.0 - momentum)
        mean_out[tile_c] = mean
    return mean_out, rstd_out


@helion.kernel(static_shapes=True)
def bn_add_bn_relu(
    x0: torch.Tensor,
    mean0: torch.Tensor,
    rstd0: torch.Tensor,
    weight0: torch.Tensor,
    bias0: torch.Tensor,
    x1: torch.Tensor,
    mean1: torch.Tensor,
    rstd1: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
) -> torch.Tensor:
    # x0, x1: [N, C, H, W] contiguous bf16; the rest [C] fp32.
    batch, channels, height, width = x0.shape
    hw = height * width
    flat0 = x0.view(-1)
    flat1 = x1.view(-1)
    numel = flat0.size(0)
    out = torch.empty([numel], dtype=x0.dtype, device=x0.device)
    for tile in hl.tile(numel):
        c = (tile.index // hw) % channels
        y0 = (flat0[tile].to(torch.float32) - mean0[c]) * rstd0[c]
        y0 = y0 * weight0[c] + bias0[c]
        y1 = (flat1[tile].to(torch.float32) - mean1[c]) * rstd1[c]
        y1 = y1 * weight1[c] + bias1[c]
        out[tile] = torch.relu(y0 + y1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x0, running_mean0, running_var0, weight0, bias0 = inputs[:5]
    x1, running_mean1, running_var1, weight1, bias1 = inputs[5:10]
    if not (x0.is_contiguous() and x1.is_contiguous()):
        # bn_add_bn_relu reads the flat NCHW activations; checked before the
        # combine kernels update the running statistics in place. Four shape
        # points are channels-last, the first one among them.
        raise ValueError("this port handles contiguous NCHW activations")
    n, c, h, w = x0.shape
    stats = []
    for x, running_mean, running_var in (
        (x0, running_mean0, running_var0),
        (x1, running_mean1, running_var1),
    ):
        mean_p, m2_p, weight_p = bn_welford_partials(
            x.view(n, c, h * w).permute(0, 2, 1), SPLITS
        )
        stats.append(
            bn_welford_combine_update(
                mean_p,
                m2_p,
                weight_p,
                running_mean,
                running_var,
                1e-05,
                0.1,
                1.0000398612827361,
            )
        )
    (mean0, rstd0), (mean1, rstd1) = stats
    out = bn_add_bn_relu(
        x0, mean0, rstd0, weight0, bias0, x1, mean1, rstd1, weight1, bias1
    )
    return (
        rstd0,
        rstd1,
        out.view(n, c, h, w),
        mean1.view(1, c, 1, 1),
        mean0.view(1, c, 1, 1),
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )
