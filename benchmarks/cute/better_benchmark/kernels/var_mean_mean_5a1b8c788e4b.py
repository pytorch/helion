"""better-benchmark repro var_mean_mean_5a1b8c788e4b (timm mobilenetv3_large_100 training).

A BatchNorm training forward (eps 1e-5, momentum 0.1) on a channels-last bf16
activation, followed by ReLU and the squeeze-excite global average pool.
Inductor emits four kernels:

- ``triton_red_fused_convert_element_type_var_mean_0``: the first level of a
  split ``var_mean`` over (N, H, W). The ``N*H*W`` rows of the ``[N*H*W, C]``
  view are cut into ``SPLITS`` (1024) blocks of 392 rows, and a looped Welford
  reduction writes each block's per-channel mean, m2 and weight to three fp32
  ``[SPLITS, C]`` workspaces.
- ``triton_red_fused_convert_element_type_var_mean_1``: the second level, a
  looped Welford combine of each of ``SPLITS2`` (8) blocks of 128 of those
  partials into three fp32 ``[SPLITS2, C]`` workspaces.
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_2``:
  a persistent combine of the remaining partials per channel into the mean and
  ``rsqrt(m2 / (N*H*W) + 1e-5)``, updating the running mean and running
  variance inputs in place (the graph's Bessel factor is the literal
  1.0000024912370735, 401408 / 401407, at every shape point).
- ``triton_red_fused_add_convert_element_type_mean_mul_relu_sub_unsqueeze_3``:
  a looped reduction per (n, c) over H*W: ``relu((x - mean) * rstd * weight +
  bias)`` stored in bf16, and the spatial mean of the ReLU output stored in
  bf16. Like Inductor (no ``emulate_precision_casts``), the graph's bf16
  rounding before the ReLU and before the mean stays in fp32.

Both split levels divide evenly at every shape point, so each block is a
whole dim of a view (``[SPLITS, rows / SPLITS, C]`` and ``[SPLITS2,
SPLITS / SPLITS2, C]``) and the block statistics are reductions over that dim:
a block's mean and its sum of squared deviations from that mean, and the
partials combine with the parallel (Chan) Welford formula.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split for these shapes: 1024 blocks of 392 rows (401408 rows),
# then 8 blocks of 128 of those partials.
SPLITS = 1024
SPLITS2 = 8


@helion.kernel(static_shapes=True)
def bn_welford_row_partials(
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
def bn_welford_combine_partials(
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # mean_p, m2_p, weight_p: [SPLITS2, partials_per_split, C] fp32 views of
    # the first level's Welford partials.
    splits, partials, channels = mean_p.shape
    mean_q = torch.empty([splits, channels], dtype=torch.float32, device=mean_p.device)
    m2_q = torch.empty([splits, channels], dtype=torch.float32, device=mean_p.device)
    weight_q = torch.empty(
        [splits, channels], dtype=torch.float32, device=mean_p.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        means = mean_p[tile_s, :, tile_c]
        m2s = m2_p[tile_s, :, tile_c]
        weights = weight_p[tile_s, :, tile_c]
        total = torch.sum(weights, dim=1)
        mean = torch.sum(means * weights, dim=1) / total
        delta = means - mean[:, None, :]
        mean_q[tile_s, tile_c] = mean
        m2_q[tile_s, tile_c] = torch.sum(m2s + weights * delta * delta, dim=1)
        weight_q[tile_s, tile_c] = total
    return mean_q, m2_q, weight_q


@helion.kernel(static_shapes=True)
def bn_stats_running_update(
    mean_q: torch.Tensor,
    m2_q: torch.Tensor,
    weight_q: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    unbias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # mean_q, m2_q, weight_q: [SPLITS2, C] fp32 partials; running_*: [C] fp32,
    # updated in place.
    splits, channels = mean_q.shape
    mean_out = torch.empty([channels], dtype=torch.float32, device=mean_q.device)
    rstd_out = torch.empty([channels], dtype=torch.float32, device=mean_q.device)
    for tile_c in hl.tile(channels):
        means = mean_q[:, tile_c]
        m2s = m2_q[:, tile_c]
        weights = weight_q[:, tile_c]
        total = torch.sum(weights, dim=0)
        mean = torch.sum(means * weights, dim=0) / total
        delta = means - mean[None, :]
        m2 = torch.sum(m2s + weights * delta * delta, dim=0)
        # The weights sum to N*H*W exactly: Inductor's true division by that
        # count (a Python constant divisor would become a reciprocal multiply).
        var = m2 / total
        rstd_out[tile_c] = torch.rsqrt(var + 1e-05)
        running_var[tile_c] = var * unbias * 0.1 + running_var[tile_c] * 0.9
        running_mean[tile_c] = mean * 0.1 + running_mean[tile_c] * 0.9
        mean_out[tile_c] = mean
    return mean_out, rstd_out


@helion.kernel(static_shapes=True)
def bn_relu_spatial_mean(
    x_rows: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x_rows: [N, H*W, C] bf16, a view of the channels-last input.
    batch, hw, channels = x_rows.shape
    act = torch.empty([batch, hw, channels], dtype=x_rows.dtype, device=x_rows.device)
    pooled = torch.empty([batch, channels], dtype=x_rows.dtype, device=x_rows.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        x = x_rows[tile_n, :, tile_c].to(torch.float32)
        y = (x - mean[tile_c][None, None, :]) * rstd[tile_c][None, None, :]
        y = y * weight[tile_c][None, None, :] + bias[tile_c][None, None, :]
        relu = torch.relu(y)
        act[tile_n, :, tile_c] = relu.to(act.dtype)
        pooled[tile_n, tile_c] = torch.mean(relu, dim=1).to(pooled.dtype)
    return act, pooled


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = inputs[:5]
    n, c, h, w = x.shape
    rows = n * h * w
    x_nhwc = x.permute(0, 2, 3, 1)
    mean_p, m2_p, weight_p = bn_welford_row_partials(
        x_nhwc.view(SPLITS, rows // SPLITS, c)
    )
    mean_q, m2_q, weight_q = bn_welford_combine_partials(
        mean_p.view(SPLITS2, SPLITS // SPLITS2, c),
        m2_p.view(SPLITS2, SPLITS // SPLITS2, c),
        weight_p.view(SPLITS2, SPLITS // SPLITS2, c),
    )
    # The graph's unbiased-variance factor is the literal 401408 / 401407.
    mean, rstd = bn_stats_running_update(
        mean_q,
        m2_q,
        weight_q,
        running_mean,
        running_var,
        1.0000024912370735,
    )
    act, pooled = bn_relu_spatial_mean(
        x_nhwc.view(n, h * w, c), mean, rstd, weight, bias
    )
    return (
        mean.view(1, c, 1, 1),
        rstd.view(1, c, 1, 1),
        act.view(n, h, w, c).permute(0, 3, 1, 2),
        pooled.view(n, c, 1, 1),
        running_mean,
        running_var,
    )
