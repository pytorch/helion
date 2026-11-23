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
over that dim, and the partials combine with the n-way (Chan) formula.

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
half ulp is a larger share of the result. No statistic is taken in fp32: an
fp32 mean of a sum carries the bias of the count's rounded reciprocal (fp32
``/`` multiplies by it on both backends, as in Inductor's Triton; 1.8 times
Inductor's error on ReLU outputs at this point), and an fp32 sum of squared
deviations rounds one way on bf16 data, whose squares are exact numbers with
biased low bits (rstd up to 9 times Inductor's error, 5 ulp, at a config that
keeps one accumulator per block). A channel that holds an infinity or a NaN
gets NaN for the mean, rstd and both running statistics, which is what
Inductor's recurrence returns (inf - inf), and ``torch.var_mean`` too unless
the infinity is the last element it folds: its block stores a NaN sum and m2
(the block's stored mean is infinite when the infinity sits outside the pivot
row). Finite values give a finite mean and rstd.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split of the N*H*W rows for this shape point: 49 blocks of 128
# rows (6272 rows).
SPLITS = 49
# The graph's scalars as the fp32 constants they are in its fp32 ops, in eager
# on CUDA and in Inductor's kernels alike: eps, the momentum weights of the
# batch and of the running statistics, and the unbiased-variance factor (the
# graph's literal, 6272 / 6271). The last statistics kernel computes with them
# in fp64, where the literals would be other numbers (fp64's 0.9 is 0.4 fp32
# ulps above fp32's).
EPS, MOMENTUM, KEEP, UNBIAS = torch.tensor(
    [1e-05, 0.1, 0.9, 1.0001594642002871], dtype=torch.float32
).tolist()


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [SPLITS, rows_per_split, C] bf16, a view of the channels-last input.
    splits, rows, channels = x_rows.shape
    sum_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    mean_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_rows.device
    )
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
        sum_p[tile_s, tile_c] = torch.where(torch.isnan(spread), spread, block_sum)
        # The block's mean rides along: the combines take the block means'
        # deviations from it, without a division per block.
        mean_p[tile_s, tile_c] = pivot + offset
        m2_p[tile_s, tile_c] = shifted_sq - shifted_sum * offset
        weight_p[tile_s, tile_c] = hl.full([tile_s, tile_c], rows, torch.float32)
    return sum_p, mean_p, m2_p, weight_p


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
    # sum_p, mean_p, m2_p: [SPLITS, C] fp64 partials, weight_p their fp32 counts;
    # running_*: [C] fp32, updated in place.
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
        sum_p, mean_p, m2_p, weight_p = bn_welford_partials(x_rows)
        stats.append(
            bn_welford_combine_update(
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
