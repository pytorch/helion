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
  output before the mean stay in fp32. The port keeps this fusion at every
  shape point. At 128x32x112x112 (bf1cc8fb) Inductor stores the SiLU output
  in one kernel and averages the stored bf16 values in two more, so its
  pooled mean follows the graph's rounding there and the port's, the
  correctly rounded mean of the unrounded values, differs from it by that
  rounding (0.4 to 1.0 bf16 ulp on average, biased where values repeat).

The split blocks are not whole tensor dims (the last block of each level is
partial), so the two split kernels loop over a block's rows with an explicit
``hl.tile`` and masked loads, adding each chunk's sums to the block's.

The batch statistics are not computed the way Inductor's kernels compute
them. Inductor's reduction kernels run Welford recurrences in fp32
(``welford_reduce`` per element in lanes, ``welford_combine`` trees over the
lanes and the partials). The port takes a block's moments in one pass and in
fp64: the sum of the rows' deviations from the block's first row (the pivot)
and the sum of their squares. For bf16 values these deviations and their
squares are exact in fp64 (while the block's values stay within a factor
2**17 of the pivot), and so are their sums while they fit 53 bits; past that,
and for other dtypes, they round at 2**-53. The chunks that the loop cuts a
block into add their sums to the block's, without a fold. The block's sum
(the pivot times its count plus the summed deviations), its mean (the pivot
plus the mean deviation) and its m2 (``sum_sq - sum * sum / count``, whose
cancellation is harmless about a pivot that is one of the block's values) go
on in fp64 workspaces: partials rounded to fp32 add their own roundings to
the result, several ulps of the mean or of rstd where blocks have few rows.
The second level adds the block sums of each of its blocks (exactly) and
combines their second moments the same way; the last kernel adds the block
sums and divides once for the mean, takes the second moment about the first
block's mean from the block means' deviations (the stored means: no division
per block), and computes the variance and ``rstd = rsqrt(var + eps)`` in fp64
as well, so each statistic it returns is rounded once, to fp32, from an fp64
value: in fp32, ``torch.rsqrt`` is the GPU's approximate reciprocal square
root on both backends and in Inductor's kernel, off by more than half an ulp
in one direction over ranges of its argument (in fp64 it measured within one
fp64 ulp on both). The running statistics' momentum updates are fp64 too (the
variance's from the fp64 variance, the mean's from the stored mean), with the
graph's scalars as the fp32 constants they are in the graph, and rounded
once: in fp32 the update rounds two or three times, depending on which
product a compiler contracts into the sum (1.3 times Inductor's error where
that is the batch statistic's). The running mean's update reads the stored
fp32 mean, the tensor the graph's update reads: where ``0.1 * mean`` and
``0.9 * running_mean`` cancel, that mean's half ulp is a larger share of the
result. No statistic is taken in fp32: an fp32 mean of a sum carries the bias
of the count's rounded reciprocal (fp32 ``/`` multiplies by it on both
backends, as in Inductor's Triton), an fp32 sum of squared deviations rounds
one way on bf16 data, whose squares are exact numbers with biased low bits,
and a block's statistics folded chunk by chunk in fp32 depend on the chunk
size (rstd 2.9 times Inductor's error on inputs with outliers at chunks of 16
rows). A channel that holds an infinity or a NaN gets NaN for the mean, rstd
and both running statistics, which is what Inductor's recurrence returns (inf
- inf), and ``torch.var_mean`` too unless the infinity is the last element it
folds: its block stores a NaN sum and m2 (the block's stored mean is infinite
when the infinity sits outside the pivot row). Finite values give a finite
mean and rstd.

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
# The graph's scalars as the fp32 constants they are in its fp32 ops, in eager
# on CUDA and in Inductor's kernels alike: eps, the momentum weights of the
# batch and of the running statistics, and the unbiased-variance factor (the
# graph's literal, 6272 / 6271, from the captured 128x7x7 activation, at every
# shape point). The last statistics kernel computes with them in fp64, where
# the literals would be other numbers (fp64's 0.9 is 0.4 fp32 ulps above
# fp32's).
EPS, MOMENTUM, KEEP, UNBIAS = torch.tensor(
    [0.001, 0.1, 0.9, 1.0001594642002871], dtype=torch.float32
).tolist()


@helion.kernel(static_shapes=True)
def bn_welford_row_partials(
    x_flat: torch.Tensor, rows_per_split: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_flat: [N*H*W, C] bf16, a view of the channels-last input.
    rows, channels = x_flat.shape
    splits = (rows + rows_per_split - 1) // rows_per_split
    sum_p = torch.empty([splits, channels], dtype=torch.float64, device=x_flat.device)
    mean_p = torch.empty([splits, channels], dtype=torch.float64, device=x_flat.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float64, device=x_flat.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_flat.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        # The block's moments about its first row, in fp64: the deviations of
        # bf16 values and their squares are exact there, and so are their sums
        # while they fit 53 bits, whatever chunks the block is cut into.
        first = tile_s.index * rows_per_split
        pivot = hl.load(x_flat, [first, tile_c], extra_mask=(first < rows)[:, None]).to(
            torch.float64
        )
        shifted_sum = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        shifted_sq = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        for tile_r in hl.tile(rows_per_split):
            row = first[:, None] + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            values = hl.load(x_flat, [row, tile_c], extra_mask=valid[:, :, None]).to(
                torch.float64
            )
            # A masked load reads zeros: the deviations are selected, so
            # that a masked lane adds nothing to either sum.
            shifted = torch.where(valid[:, :, None], values - pivot[:, None, :], 0.0)
            shifted_sum = shifted_sum + torch.sum(shifted, dim=1)
            shifted_sq = shifted_sq + torch.sum(shifted * shifted, dim=1)
        # The block's rows: rows_per_split, fewer in the last block.
        count = torch.clamp(rows - first, max=rows_per_split).to(torch.float64)[:, None]
        offset = shifted_sum / count
        block_sum = pivot * count + shifted_sum
        # A block that holds an infinity has an infinite or NaN sum here and
        # a NaN m2 (inf - inf); its sum is stored as NaN, as Inductor's
        # recurrence returns the mean.
        spread = block_sum - block_sum
        sum_p[tile_s, tile_c] = torch.where(torch.isnan(spread), spread, block_sum)
        # The block's mean rides along: the combines take the block means'
        # deviations from it, without a division per block.
        mean_p[tile_s, tile_c] = pivot + offset
        m2_p[tile_s, tile_c] = shifted_sq - shifted_sum * offset
        weight_p[tile_s, tile_c] = hl.zeros(
            [tile_s, tile_c], dtype=torch.float32
        ) + count.to(torch.float32)
    return sum_p, mean_p, m2_p, weight_p


@helion.kernel(static_shapes=True)
def bn_welford_combine_partials(
    sum_p: torch.Tensor,
    mean_p: torch.Tensor,
    m2_p: torch.Tensor,
    weight_p: torch.Tensor,
    partials_per_split: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # sum_p, mean_p, m2_p: [partials, C] fp64 partials, weight_p their fp32
    # counts.
    partials, channels = sum_p.shape
    splits = (partials + partials_per_split - 1) // partials_per_split
    sum_q = torch.empty([splits, channels], dtype=torch.float64, device=sum_p.device)
    mean_q = torch.empty([splits, channels], dtype=torch.float64, device=sum_p.device)
    m2_q = torch.empty([splits, channels], dtype=torch.float64, device=sum_p.device)
    weight_q = torch.empty([splits, channels], dtype=torch.float32, device=sum_p.device)
    for tile_s, tile_c in hl.tile([splits, channels]):
        # The n-way (Chan) combine in fp64: the sums add exactly, and the
        # second moment is taken about the block's first partial mean, a
        # pivot again, from the partial means' deviations.
        first = tile_s.index * partials_per_split
        pivot = hl.load(mean_p, [first, tile_c], extra_mask=(first < partials)[:, None])
        total = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        total_sum = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        shifted_sum = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        m2 = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        for tile_r in hl.tile(partials_per_split):
            idx = first[:, None] + tile_r.index[None, :]
            valid = (idx < partials) & (tile_r.index[None, :] < partials_per_split)
            mask = valid[:, :, None]
            # A masked load reads zeros: a masked lane has no weight, and
            # adds nothing to any of the sums.
            sums = hl.load(sum_p, [idx, tile_c], extra_mask=mask)
            means = hl.load(mean_p, [idx, tile_c], extra_mask=mask)
            m2s = hl.load(m2_p, [idx, tile_c], extra_mask=mask)
            weights = hl.load(weight_p, [idx, tile_c], extra_mask=mask).to(
                torch.float64
            )
            shifted = means - pivot[:, None, :]
            weighted = weights * shifted
            total = total + torch.sum(weights, dim=1)
            total_sum = total_sum + torch.sum(sums, dim=1)
            shifted_sum = shifted_sum + torch.sum(weighted, dim=1)
            m2 = m2 + torch.sum(m2s + weighted * shifted, dim=1)
        sum_q[tile_s, tile_c] = total_sum
        mean_q[tile_s, tile_c] = total_sum / total
        m2_q[tile_s, tile_c] = (m2 * total - shifted_sum * shifted_sum) / total
        weight_q[tile_s, tile_c] = total.to(torch.float32)
    return sum_q, mean_q, m2_q, weight_q


@helion.kernel(static_shapes=True)
def bn_stats_running_update(
    sum_q: torch.Tensor,
    mean_q: torch.Tensor,
    m2_q: torch.Tensor,
    weight_q: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: hl.constexpr,
    momentum: hl.constexpr,
    keep: hl.constexpr,
    unbias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sum_q, mean_q, m2_q: [splits, C] fp64 partials, weight_q their fp32
    # counts; running_*: [C] fp32, updated in place.
    splits, channels = sum_q.shape
    mean_out = torch.empty([channels], dtype=torch.float32, device=sum_q.device)
    rstd_out = torch.empty([channels], dtype=torch.float32, device=sum_q.device)
    for tile_c in hl.tile(channels):
        # The n-way (Chan) combine in fp64. The mean is the sum of the block
        # sums (exact, like them) over the count; the second moment is taken
        # about the first block's mean, a pivot again, from the block means'
        # deviations. The mean, the variance and rstd round once, to fp32.
        weights = weight_q[:, tile_c].to(torch.float64)
        total = torch.sum(weights, dim=0)
        mean64 = torch.sum(sum_q[:, tile_c], dim=0) / total
        pivot = mean_q[0, tile_c]
        shifted = mean_q[:, tile_c] - pivot[None, :]
        shifted_sum = torch.sum(weights * shifted, dim=0)
        m2 = torch.sum(m2_q[:, tile_c] + weights * shifted * shifted, dim=0)
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
    sum_p, mean_p, m2_p, weight_p = bn_welford_row_partials(
        x_nhwc.view(rows, c), ROWS_PER_SPLIT
    )
    sum_q, mean_q, m2_q, weight_q = bn_welford_combine_partials(
        sum_p, mean_p, m2_p, weight_p, PARTIALS_PER_SPLIT
    )
    mean, rstd = bn_stats_running_update(
        sum_q,
        mean_q,
        m2_q,
        weight_q,
        running_mean,
        running_var,
        EPS,
        MOMENTUM,
        KEEP,
        UNBIAS,
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
