"""better-benchmark repro var_mean_var_mean_mean_0efb69a14422 (timm repvgg_a2 training).

The last RepVGG block's two BatchNorm training forwards (the 3x3 and 1x1
branches; eps 1e-5, momentum 0.1) on channels-last bf16 activations, each
updating its running mean and running variance inputs in place, whose affine
outputs are added and passed through ReLU, followed by the head's global
average pool. Inductor emits five kernel launches, three distinct kernels:

- ``triton_red_fused_convert_element_type_var_mean_0`` (once per input): the
  first level of a split ``var_mean`` over (N, H, W). The ``N*H*W`` rows of
  the ``[N*H*W, C]`` view are cut into ``SPLITS`` (49) blocks of 128
  consecutive rows, and a looped Welford reduction writes each block's
  per-channel mean, m2 and weight to three fp32 ``[SPLITS, C]`` workspaces.
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_1``
  (once per input): a persistent combine of the partials per channel into the
  mean and ``rsqrt(m2 / (N*H*W) + 1e-5)``, updating the running statistics in
  place (the graph's Bessel factor is the literal 1.0001594642002871, 6272 /
  6271).
- ``triton_per_fused_add_convert_element_type_mean_mul_relu_sub_unsqueeze_2``:
  a persistent reduction per (n, c) over H*W of ``relu(((x0 - mean0) * rstd0
  * weight0 + bias0) + ((x1 - mean1) * rstd1 * weight1 + bias1))``, storing
  only the spatial mean in bf16 (the ReLU output itself is not written). Like
  Inductor (no ``emulate_precision_casts``), the graph's bf16 rounding of each
  affine output and of their sum (the ReLU input) stays in fp32; only the
  pooled store rounds.

The split divides the rows evenly, so each block is a whole dim of the
``[SPLITS, rows / SPLITS, C]`` view and the block statistics are reductions
over that dim: a block's mean and its sum of squared deviations from that
mean, and the partials combine with the parallel (Chan) Welford formula.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split of the N*H*W rows for this shape point: 49 blocks of 128
# rows (6272 rows).
SPLITS = 49


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [SPLITS, rows_per_split, C] bf16, a view of the channels-last input.
    splits, rows, channels = x_rows.shape
    mean_p = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_rows.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        values = x_rows[tile_s, :, tile_c].to(torch.float32)
        block_mean = torch.mean(values, dim=1)
        centered = values - block_mean[:, None, :]
        mean_p[tile_s, tile_c] = block_mean
        m2_p[tile_s, tile_c] = torch.sum(centered * centered, dim=1)
        weight_p[tile_s, tile_c] = hl.full([tile_s, tile_c], rows, torch.float32)
    return mean_p, m2_p, weight_p


@helion.kernel(static_shapes=True)
def bn_welford_combine_update(
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # mean_p, m2_p, weight_p: [SPLITS, C] fp32 partials; running_*: [C] fp32,
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
        # count (a Python constant divisor would become a reciprocal multiply).
        var = m2 / total
        rstd_out[tile_c] = torch.rsqrt(var + 1e-05)
        running_var[tile_c] = var * 1.0001594642002871 * 0.1 + running_var[tile_c] * 0.9
        running_mean[tile_c] = mean * 0.1 + running_mean[tile_c] * 0.9
        mean_out[tile_c] = mean
    return mean_out, rstd_out


@helion.kernel(static_shapes=True)
def bn_add_bn_relu_spatial_mean(
    x0_rows: torch.Tensor,
    mean0: torch.Tensor,
    rstd0: torch.Tensor,
    weight0: torch.Tensor,
    bias0: torch.Tensor,
    x1_rows: torch.Tensor,
    mean1: torch.Tensor,
    rstd1: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
) -> torch.Tensor:
    # x0_rows, x1_rows: [N, H*W, C] bf16 views of the channels-last inputs;
    # the rest [C] fp32.
    batch, hw, channels = x0_rows.shape
    pooled = torch.empty([batch, channels], dtype=x0_rows.dtype, device=x0_rows.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        y0 = (
            x0_rows[tile_n, :, tile_c].to(torch.float32) - mean0[tile_c][None, None, :]
        ) * rstd0[tile_c][None, None, :]
        y0 = y0 * weight0[tile_c][None, None, :] + bias0[tile_c][None, None, :]
        y1 = (
            x1_rows[tile_n, :, tile_c].to(torch.float32) - mean1[tile_c][None, None, :]
        ) * rstd1[tile_c][None, None, :]
        y1 = y1 * weight1[tile_c][None, None, :] + bias1[tile_c][None, None, :]
        relu = torch.relu(y0 + y1)
        pooled[tile_n, tile_c] = torch.mean(relu, dim=1).to(pooled.dtype)
    return pooled


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x0, running_mean0, running_var0, weight0, bias0 = inputs[:5]
    x1, running_mean1, running_var1, weight1, bias1 = inputs[5:10]
    # [N, C, 1, 1], its strides, and [N, C]: the pooled output's as_strided
    # and view.
    pool_size, pool_stride, pool_view = inputs[10:13]
    n, c, h, w = x0.shape
    rows = n * h * w
    x0_nhwc = x0.permute(0, 2, 3, 1)
    x1_nhwc = x1.permute(0, 2, 3, 1)
    # Every view before the first launch: a layout they reject must fail
    # before the combines update the running statistics in place.
    split_rows = [x.view(SPLITS, rows // SPLITS, c) for x in (x0_nhwc, x1_nhwc)]
    x0_rows, x1_rows = (x.view(n, h * w, c) for x in (x0_nhwc, x1_nhwc))
    stats = []
    for x_rows, running_mean, running_var in zip(
        split_rows,
        (running_mean0, running_mean1),
        (running_var0, running_var1),
        strict=True,
    ):
        mean_p, m2_p, weight_p = bn_welford_partials(x_rows)
        stats.append(
            bn_welford_combine_update(mean_p, m2_p, weight_p, running_mean, running_var)
        )
    (mean0, rstd0), (mean1, rstd1) = stats
    pooled = bn_add_bn_relu_spatial_mean(
        x0_rows,
        mean0,
        rstd0,
        weight0,
        bias0,
        x1_rows,
        mean1,
        rstd1,
        weight1,
        bias1,
    )
    return (
        mean0.view(1, c, 1, 1),
        rstd0.view(1, c, 1, 1),
        mean1.view(1, c, 1, 1),
        rstd1.view(1, c, 1, 1),
        pooled.as_strided(pool_size, pool_stride).view(pool_view),
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )
