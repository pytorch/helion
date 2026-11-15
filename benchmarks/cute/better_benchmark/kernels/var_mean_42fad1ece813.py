"""better-benchmark repro var_mean_42fad1ece813 (timm inception_v3 training).

BatchNorm training forward on a channels-last bf16 activation: per-channel
``var_mean`` over (N, H, W), running-stat updates written into the running
mean/var inputs (momentum 0.1; the variance is unbiased with the graph's
literal factor 1.0001220852154804, 8192 / 8191 from the captured 128x8x8
activation, at every shape point), and the affine normalize followed by
ReLU. At the two channels-last 8192-row shape points (652894ae, 8e6a7889)
Inductor emits three kernels, which the port reproduces (the other
channels-last points compile to four):

- ``triton_red_fused_convert_element_type_var_mean_0``: split reduction; each
  of ``splits`` row blocks writes per-channel Welford partials (mean, m2,
  weight) to three fp32 ``[splits, C]`` workspaces.
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_1``:
  combines the partials per channel into mean and rstd, dividing by the
  summed weights as Inductor's Welford combine divides by its count, and
  updates the running statistics in place.
- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_2``:
  ``relu((x - mean) * rstd * weight + bias)`` in bf16.

The kernels read the ``[N*H*W, C]`` view of a channels-last input, and the
split views it as ``[64, N*H*W / 64, C]``: N*H*W is a multiple of 64 at all
26 channels-last shape points. ``forward()`` rejects the 95 NCHW points
before its first launch, which would update the running statistics.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [splits, rows_per_split, C], a view of the channels-last input.
    splits, rows, channels = x_rows.shape
    mean = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    m2 = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    weight = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    for tile_s, tile_c in hl.tile([splits, channels]):
        values = x_rows[tile_s, :, tile_c].to(torch.float32)
        block_mean = torch.mean(values, dim=1)
        centered = values - block_mean[:, None, :]
        mean[tile_s, tile_c] = block_mean
        m2[tile_s, tile_c] = torch.sum(centered * centered, dim=1)
        weight[tile_s, tile_c] = hl.full([tile_s, tile_c], rows, torch.float32)
    return mean, m2, weight


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
        # count (a constant divisor would become a reciprocal multiply).
        var = m2 / total
        rstd_out[tile_c] = torch.rsqrt(var + eps)
        running_var[tile_c] = var * unbias * momentum + running_var[tile_c] * (
            1.0 - momentum
        )
        running_mean[tile_c] = mean * momentum + running_mean[tile_c] * (1.0 - momentum)
        mean_out[tile_c] = mean
    return mean_out, rstd_out


@helion.kernel(static_shapes=True)
def bn_normalize_relu(
    x_flat: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # x_flat: [N*H*W, C], a view of the channels-last input.
    rows, channels = x_flat.shape
    out = torch.empty([rows, channels], dtype=x_flat.dtype, device=x_flat.device)
    for tile_r, tile_c in hl.tile([rows, channels]):
        y = (x_flat[tile_r, tile_c].to(torch.float32) - mean[tile_c]) * rstd[tile_c]
        y = y * weight[tile_c] + bias[tile_c]
        out[tile_r, tile_c] = torch.relu(y.to(out.dtype))
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = inputs[:5]
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("this port handles channels-last activations")
    n, c, h, w = x.shape
    rows = n * h * w
    x_flat = x.permute(0, 2, 3, 1).view(rows, c)
    # Inductor's split at the 8192-row points: 64 row blocks of rows / 64.
    splits = 64
    mean_p, m2_p, weight_p = bn_welford_partials(x_flat.view(splits, rows // splits, c))
    mean, rstd = bn_welford_combine_update(
        mean_p,
        m2_p,
        weight_p,
        running_mean,
        running_var,
        0.001,
        0.1,
        1.0001220852154804,
    )
    out = bn_normalize_relu(x_flat, mean, rstd, weight, bias)
    out = out.view(n, h, w, c).permute(0, 3, 1, 2)
    return rstd, out, mean.view(1, c, 1, 1), running_mean, running_var
