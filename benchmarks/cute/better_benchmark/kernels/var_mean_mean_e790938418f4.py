"""better-benchmark repro var_mean_mean_e790938418f4 (timm/train/tf_efficientnet_b0).

A BatchNorm training forward (eps 1e-3, momentum 0.1) on a channels-last bf16
activation, followed by SiLU and the squeeze-excite global average pool.
Inductor emits four kernels:

- ``triton_red_fused_convert_element_type_var_mean_0``: the first level of a
  split ``var_mean`` over (N, H, W). The ``N*H*W`` rows of the
  ``[N*H*W, C]`` view are cut into blocks of ``rows_per_split`` (680) rows,
  the last one partial, and a looped Welford reduction writes each block's
  per-channel mean, m2 and weight to three fp32 ``[splits, C]`` workspaces.
- ``triton_red_fused_convert_element_type_var_mean_1``: the second level, a
  looped Welford combine of blocks of ``partials_per_split`` (119) of those
  partials (the last block partial) into three fp32 ``[splits2, C]``
  workspaces.
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_2``:
  a persistent combine of the remaining partials per channel into the mean and
  ``rsqrt(m2 / (N*H*W) + 1e-3)``, updating the running mean and running
  variance inputs in place (the graph's Bessel factor is the literal
  1.0001594642002871 at every shape point).
- ``triton_red_fused_add_convert_element_type_div_exp_mean_mul_neg_sub_unsqueeze_3``:
  a looped reduction per (n, c) over H*W: ``y = (x - mean) * rstd * weight +
  bias`` and ``y / (exp(-y) + 1)`` stored in bf16, and the spatial mean of the
  SiLU output stored in bf16. Like Inductor (no ``emulate_precision_casts``),
  the graph's bf16 round trip of ``y`` and the bf16 rounding of the SiLU
  output before the mean stay in fp32.

The split blocks are not whole tensor dims (the last block of each level is
partial), so the two split kernels loop over a block's rows with an explicit
``hl.tile`` and masked loads, combining each chunk's statistics into running
Welford accumulators, as Inductor's looped reductions do.

When this port was made, CuTe declined some configs of
``bn_welford_row_partials`` at codegen for want of shared memory
(``BackendUnsupported``): with the row axis over several warps and vector
widths that unroll the split and channel lanes, each unrolled copy of a
chunk reduction took its own shared-memory buffer.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split for this shape: 591 blocks of 680 rows (401408 rows), then
# 5 blocks of 119 of those partials.
ROWS_PER_SPLIT = 680
PARTIALS_PER_SPLIT = 119


@helion.kernel(static_shapes=True)
def bn_welford_row_partials(
    x_flat: torch.Tensor, rows_per_split: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_flat: [N*H*W, C] bf16, a view of the channels-last input.
    rows, channels = x_flat.shape
    splits = (rows + rows_per_split - 1) // rows_per_split
    mean_p = torch.empty([splits, channels], dtype=torch.float32, device=x_flat.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float32, device=x_flat.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_flat.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        mean = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        m2 = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        weight = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            values = hl.load(x_flat, [row, tile_c], extra_mask=valid[:, :, None]).to(
                torch.float32
            )
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
def bn_welford_combine_partials(
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
    partials_per_split: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # mean_p, m2_p, weight_p: [partials, C] fp32 Welford partials.
    partials, channels = mean_p.shape
    splits = (partials + partials_per_split - 1) // partials_per_split
    mean_q = torch.empty([splits, channels], dtype=torch.float32, device=mean_p.device)
    m2_q = torch.empty([splits, channels], dtype=torch.float32, device=mean_p.device)
    weight_q = torch.empty(
        [splits, channels], dtype=torch.float32, device=mean_p.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        mean = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        m2 = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        weight = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(partials_per_split):
            idx = tile_s.index[:, None] * partials_per_split + tile_r.index[None, :]
            valid = (idx < partials) & (tile_r.index[None, :] < partials_per_split)
            means = hl.load(mean_p, [idx, tile_c], extra_mask=valid[:, :, None])
            m2s = hl.load(m2_p, [idx, tile_c], extra_mask=valid[:, :, None])
            weights = hl.load(weight_p, [idx, tile_c], extra_mask=valid[:, :, None])
            chunk_weight = torch.sum(weights, dim=1)
            chunk_mean = torch.where(
                chunk_weight > 0.0,
                torch.sum(means * weights, dim=1) / chunk_weight,
                0.0,
            )
            spread = means - chunk_mean[:, None, :]
            chunk_m2 = torch.sum(m2s + weights * spread * spread, dim=1)
            new_weight = weight + chunk_weight
            ratio = torch.where(new_weight == 0.0, 0.0, chunk_weight / new_weight)
            delta = chunk_mean - mean
            mean = mean + delta * ratio
            m2 = m2 + chunk_m2 + delta * delta * weight * ratio
            weight = new_weight
        mean_q[tile_s, tile_c] = mean
        m2_q[tile_s, tile_c] = m2
        weight_q[tile_s, tile_c] = weight
    return mean_q, m2_q, weight_q


@helion.kernel(static_shapes=True)
def bn_stats_running_update(
    mean_q: torch.Tensor,
    m2_q: torch.Tensor,
    weight_q: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: hl.constexpr,
    momentum: hl.constexpr,
    unbias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # mean_q, m2_q, weight_q: [splits, C] fp32 partials; running_*: [C] fp32,
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
        # The weights sum to N*H*W exactly: Inductor's division by that count.
        var = m2 / total
        rstd_out[tile_c] = torch.rsqrt(var + eps)
        running_var[tile_c] = var * unbias * momentum + running_var[tile_c] * (
            1.0 - momentum
        )
        running_mean[tile_c] = mean * momentum + running_mean[tile_c] * (1.0 - momentum)
        mean_out[tile_c] = mean
    return mean_out, rstd_out


@helion.kernel(static_shapes=True)
def bn_silu_spatial_mean(
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
        silu = y / (torch.exp(-y) + 1.0)
        act[tile_n, :, tile_c] = silu.to(act.dtype)
        pooled[tile_n, tile_c] = torch.mean(silu, dim=1).to(pooled.dtype)
    return act, pooled


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = inputs[:5]
    n, c, h, w = x.shape
    rows = n * h * w
    x_nhwc = x.permute(0, 2, 3, 1)
    mean_p, m2_p, weight_p = bn_welford_row_partials(
        x_nhwc.view(rows, c), ROWS_PER_SPLIT
    )
    mean_q, m2_q, weight_q = bn_welford_combine_partials(
        mean_p, m2_p, weight_p, PARTIALS_PER_SPLIT
    )
    # The graph's unbiased-variance factor is a literal, 6272 / 6271 (from the
    # captured 128x7x7 activation), at every shape point.
    mean, rstd = bn_stats_running_update(
        mean_q,
        m2_q,
        weight_q,
        running_mean,
        running_var,
        0.001,
        0.1,
        1.0001594642002871,
    )
    act, pooled = bn_silu_spatial_mean(
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
