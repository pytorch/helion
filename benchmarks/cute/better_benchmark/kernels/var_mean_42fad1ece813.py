"""better-benchmark repro var_mean_42fad1ece813 (timm inception_v3 training).

BatchNorm training forward on a channels-last bf16 activation: per-channel
``var_mean`` over (N, H, W), running-stat updates written into the running
mean/var inputs (momentum 0.1; the variance is unbiased with the graph's
literal factor 1.0001220852154804, 8192 / 8191 from the captured 128x8x8
activation, at every shape point), and the affine normalize followed by
ReLU. At the two channels-last 8192-row shape points (652894ae, 8e6a7889)
Inductor emits three kernels, and the port has one for each (the other
channels-last points compile to four):

- ``triton_red_fused_convert_element_type_var_mean_0``: split reduction; each
  of ``splits`` row blocks writes per-channel Welford partials (mean, m2,
  weight) to three fp32 ``[splits, C]`` workspaces. The port's kernel writes
  four: the block's sum, mean and m2 in fp64 and its count in fp32 (below).
- ``triton_per_fused_add_convert_element_type_copy__mul_rsqrt_squeeze_var_mean_1``:
  combines the partials per channel into mean and rstd and updates the
  running statistics in place.
- ``triton_poi_fused_add_convert_element_type_mul_relu_sub_unsqueeze_2``:
  ``relu((x - mean) * rstd * weight + bias)`` in bf16.

The kernels read the ``[N*H*W, C]`` view of a channels-last input, and the
split views it as ``[64, N*H*W / 64, C]``: N*H*W is a multiple of 64 at all
26 channels-last shape points. ``forward()`` rejects the 95 NCHW points
before its first launch, which would update the running statistics.

The batch statistics are not computed the way Inductor's kernels compute
them. Inductor's reduction kernels run Welford recurrences in fp32
(``welford_reduce`` per element in lanes, ``welford_combine`` trees over the
lanes and the partials). The port takes a block's moments in one pass and in
fp64: the sum of the rows' deviations from the block's first row (the pivot)
and the sum of their squares. For bf16 values these deviations and their
squares are exact in fp64 (while the block's values stay within a factor
2**17 of the pivot), and so are their sums while they fit 53 bits; past that,
and for other dtypes, they round at 2**-53. The block's sum (the pivot times
its count plus the summed deviations), its mean (the pivot plus the mean
deviation) and its m2 (``sum_sq - sum * sum / count``, whose cancellation is
harmless about a pivot that is one of the block's values) go on in fp64
workspaces: partials rounded to fp32 add their own roundings to the result,
several ulps of the mean or of rstd where blocks have few rows. The combine
adds the block sums and divides once for the mean, takes the second moment
about the first block's mean from the block means' deviations (the stored
means: no division per block), and computes the variance and ``rstd =
rsqrt(var + eps)`` in fp64 as well, so each statistic it returns is rounded
once, to fp32, from an fp64 value: in fp32, ``torch.rsqrt`` is the GPU's
approximate reciprocal square root on both backends and in Inductor's kernel,
off by more than half an ulp in one direction over ranges of its argument (in
fp64 it measured within one fp64 ulp on both). The running statistics'
momentum updates are fp64 too (the variance's from the fp64 variance, the
mean's from the stored mean), with the graph's scalars as the fp32 constants
they are in the graph, and rounded once: in fp32 the update rounds two or
three times, depending on which product a compiler contracts into the sum
(1.3 times Inductor's error where that is the batch statistic's). The running
mean's update reads the stored fp32 mean, the tensor the graph's update
reads: where ``0.1 * mean`` and ``0.9 * running_mean`` cancel, that mean's
half ulp is a larger share of the result. No statistic is taken in fp32,
although this point's counts are powers of two (128 rows per block, 8192
rows), whose fp32 quotients are exact: an fp32 sum of squared deviations
rounds one way on bf16 data, whose squares are exact numbers with biased low
bits (rstd 1.6 times Inductor's error on sparse ReLU outputs at this point's
tuned config), and for other counts an fp32 mean of a sum carries the bias of
the count's rounded reciprocal (fp32 ``/`` multiplies by it on both backends,
as in Inductor's Triton). A channel that holds an infinity or a NaN gets NaN
for the mean, rstd and both running statistics, which is what Inductor's
recurrence returns (inf - inf), and ``torch.var_mean`` too unless the
infinity is the last element it folds: its block stores a NaN sum and m2 (the
block's stored mean is infinite when the infinity sits outside the pivot
row). Finite values give a finite mean and rstd.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# The graph's scalars as the fp32 constants they are in its fp32 ops, in eager
# on CUDA and in Inductor's kernels alike: eps, the momentum weights of the
# batch and of the running statistics, and the unbiased-variance factor (the
# graph's literal, 8192 / 8191). The last statistics kernel computes with them
# in fp64, where the literals would be other numbers (fp64's 0.9 is 0.4 fp32
# ulps above fp32's).
EPS, MOMENTUM, KEEP, UNBIAS = torch.tensor(
    [0.001, 0.1, 0.9, 1.0001220852154804], dtype=torch.float32
).tolist()


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [splits, rows_per_split, C], a view of the channels-last input.
    splits, rows, channels = x_rows.shape
    block_sums = torch.empty(
        [splits, channels], dtype=torch.float64, device=x_rows.device
    )
    block_means = torch.empty(
        [splits, channels], dtype=torch.float64, device=x_rows.device
    )
    m2 = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    weight = torch.empty([splits, channels], dtype=torch.float32, device=x_rows.device)
    for tile_s, tile_c in hl.tile([splits, channels]):
        # The block's moments about its first row, in fp64: the deviations of
        # bf16 values and their squares are exact there, and so are their sums
        # while they fit 53 bits.
        pivot = x_rows[tile_s, 0, tile_c].to(torch.float64)
        shifted = x_rows[tile_s, :, tile_c].to(torch.float64) - pivot[:, None, :]
        shifted_sum = torch.sum(shifted, dim=1)
        shifted_sq = torch.sum(shifted * shifted, dim=1)
        offset = shifted_sum / rows
        block_sum = pivot * rows + shifted_sum
        # A block that holds an infinity has an infinite or NaN sum here and
        # a NaN m2 (inf - inf); its sum is stored as NaN, as Inductor's
        # recurrence returns the mean.
        spread = block_sum - block_sum
        block_sums[tile_s, tile_c] = torch.where(torch.isnan(spread), spread, block_sum)
        # The block's mean rides along: the combines take the block means'
        # deviations from it, without a division per block.
        block_means[tile_s, tile_c] = pivot + offset
        m2[tile_s, tile_c] = shifted_sq - shifted_sum * offset
        weight[tile_s, tile_c] = hl.full([tile_s, tile_c], rows, torch.float32)
    return block_sums, block_means, m2, weight


@helion.kernel(static_shapes=True)
def bn_welford_combine_update(
    sum_p: torch.Tensor,
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: hl.constexpr,
    momentum: hl.constexpr,
    keep: hl.constexpr,
    unbias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    splits, channels = sum_p.shape
    mean_out = torch.empty([channels], dtype=torch.float32, device=sum_p.device)
    rstd_out = torch.empty([channels], dtype=torch.float32, device=sum_p.device)
    for tile_c in hl.tile(channels):
        # The n-way (Chan) combine in fp64. The mean is the sum of the block
        # sums (exact, like them) over the count; the second moment is taken
        # about the first block's mean, a pivot again, from the block means'
        # deviations. The mean, the variance and rstd round once, to fp32.
        weights = weight_p[:, tile_c].to(torch.float64)
        total = torch.sum(weights, dim=0)
        mean64 = torch.sum(sum_p[:, tile_c], dim=0) / total
        pivot = mean_p[0, tile_c]
        shifted = mean_p[:, tile_c] - pivot[None, :]
        shifted_sum = torch.sum(weights * shifted, dim=0)
        m2 = torch.sum(m2_p[:, tile_c] + weights * shifted * shifted, dim=0)
        mean = mean64.to(torch.float32)
        # The weights sum to N*H*W exactly; m2 - shifted_sum**2 / total over
        # total, in one division.
        var64 = (m2 * total - shifted_sum * shifted_sum) / (total * total)
        # rstd from the fp64 variance, rounded once: fp64 rsqrt is within an
        # fp64 ulp; fp32 rsqrt is the GPU's approximation (in Inductor's too).
        rstd_out[tile_c] = torch.rsqrt(var64 + eps).to(torch.float32)
        # The momentum updates in fp64, rounded once: of the fp64 variance,
        # and of the mean as it is stored. In fp32 the two products' sum rounds
        # two or three times (Inductor's kernel: twice, the running one fused).
        running_var[tile_c] = (
            var64 * unbias * momentum + running_var[tile_c].to(torch.float64) * keep
        ).to(torch.float32)
        running_mean[tile_c] = (
            mean.to(torch.float64) * momentum
            + running_mean[tile_c].to(torch.float64) * keep
        ).to(torch.float32)
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
    sum_p, mean_p, m2_p, weight_p = bn_welford_partials(
        x_flat.view(splits, rows // splits, c)
    )
    mean, rstd = bn_welford_combine_update(
        sum_p,
        mean_p,
        m2_p,
        weight_p,
        running_mean,
        running_var,
        EPS,
        MOMENTUM,
        KEEP,
        UNBIAS,
    )
    out = bn_normalize_relu(x_flat, mean, rstd, weight, bias)
    out = out.view(n, h, w, c).permute(0, 3, 1, 2)
    return rstd, out, mean.view(1, c, 1, 1), running_mean, running_var
