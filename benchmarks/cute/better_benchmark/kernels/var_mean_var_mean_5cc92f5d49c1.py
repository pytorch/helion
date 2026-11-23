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
  (``n = r // (H*W)``, ``hw = r % (H*W)``), adding each chunk's sums to
  the block's.
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

The batch statistics are not computed the way Inductor's kernels compute
them. Inductor's reduction kernels run Welford recurrences in fp32
(``welford_reduce`` per element in lanes, ``welford_combine`` trees over the
lanes and the partials). The port takes a block's moments in one pass and in
fp64: the sum of the elements' deviations from the block's first element (the
pivot) and the sum of their squares. For bf16 values these deviations and
their squares are exact in fp64 (while the block's values stay within a
factor 2**17 of the pivot), and so are their sums while they fit 53 bits;
past that, and for other dtypes, they round at 2**-53. The chunks that the
loop cuts a block into add their sums to the block's, without a fold. The
block's sum (the pivot times its count plus the summed deviations), its mean
(the pivot plus the mean deviation) and its m2 (``sum_sq - sum * sum /
count``, whose cancellation is harmless about a pivot that is one of the
block's values) go on in fp64 workspaces: partials rounded to fp32 add their
own roundings to the result, several ulps of the mean or of rstd where blocks
have few elements. The combine adds the block sums and divides once for the
mean, takes the second moment about the first block's mean from the block
means' deviations (the stored means: no division per block), and computes the
variance and ``rstd = rsqrt(var + eps)`` in fp64 as well, so each statistic
it returns is rounded once, to fp32, from an fp64 value: in fp32,
``torch.rsqrt`` is the GPU's approximate reciprocal square root on both
backends and in Inductor's kernel, off by more than half an ulp in one
direction over ranges of its argument (in fp64 it measured within one fp64
ulp on both). The running statistics' momentum updates are fp64 too (the
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
size (rstd up to 56 times Inductor's error at chunks of 16 elements). A
channel that holds an infinity or a NaN gets NaN for the mean, rstd and both
running statistics, which is what Inductor's recurrence returns (inf - inf),
and ``torch.var_mean`` too unless the infinity is the last element it folds:
its block stores a NaN sum and m2 (the block's stored mean is infinite when
the infinity sits outside the pivot row). Finite values give a finite mean
and rstd.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's split of each channel's (N, H, W) reduction for this shape: 4
# blocks of 25088 elements (8 images of 56x56).
SPLITS = 4
# The graph's scalars as the fp32 constants they are in its fp32 ops, in eager
# on CUDA and in Inductor's kernels alike: eps, the momentum weights of the
# batch and of the running statistics, and the unbiased-variance factor (the
# graph's literal, 25088 / 25087, at every shape point). The last statistics
# kernel computes with them in fp64, where the literals would be other numbers
# (fp64's 0.9 is 0.4 fp32 ulps above fp32's).
EPS, MOMENTUM, KEEP, UNBIAS = torch.tensor(
    [1e-05, 0.1, 0.9, 1.0000398612827361], dtype=torch.float32
).tolist()


@helion.kernel(static_shapes=True)
def bn_welford_partials(
    x_rows: torch.Tensor, splits: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_rows: [N, H*W, C] bf16, a permuted view of the NCHW input.
    batch, hw, channels = x_rows.shape
    total = batch * hw
    rows_per_split = (total + splits - 1) // splits
    sum_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    mean_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    m2_p = torch.empty([splits, channels], dtype=torch.float64, device=x_rows.device)
    weight_p = torch.empty(
        [splits, channels], dtype=torch.float32, device=x_rows.device
    )
    for tile_s, tile_c in hl.tile([splits, channels]):
        # The block's moments about its first element, in fp64: the
        # deviations of bf16 values and their squares are exact there, and so
        # are their sums while they fit 53 bits, whatever chunks the block is
        # cut into.
        # first, r: indices into the channel's flattened (n, h, w) elements.
        # The pivot is loaded as a [tile_s, 1, tile_c] chunk of one element,
        # like the loop's chunks, and summed over it (rank-one index tensors
        # for the view's two row indices give a load of another rank).
        first = tile_s.index[:, None] * rows_per_split
        pivot = torch.sum(
            hl.load(
                x_rows,
                [first // hw, first % hw, tile_c],
                extra_mask=(first < total)[:, :, None],
            ).to(torch.float64),
            dim=1,
        )
        shifted_sum = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        shifted_sq = hl.zeros([tile_s, tile_c], dtype=torch.float64)
        for tile_r in hl.tile(rows_per_split):
            r = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (r < total) & (tile_r.index[None, :] < rows_per_split)
            values = hl.load(
                x_rows, [r // hw, r % hw, tile_c], extra_mask=valid[:, :, None]
            ).to(torch.float64)
            # A masked load reads zeros: the deviations are selected, so
            # that a masked lane adds nothing to either sum.
            shifted = torch.where(valid[:, :, None], values - pivot[:, None, :], 0.0)
            shifted_sum = shifted_sum + torch.sum(shifted, dim=1)
            shifted_sq = shifted_sq + torch.sum(shifted * shifted, dim=1)
        # The block's elements: rows_per_split, fewer in a partial last block
        # and none in a block past the end (which a channel of a few elements
        # can leave): such a block's statistics are zeros of zero weight.
        count = torch.clamp(total - first, min=0, max=rows_per_split).to(torch.float64)
        offset = torch.where(count > 0.0, shifted_sum / count, 0.0)
        block_sum = pivot * count + shifted_sum
        # A block that holds an infinity has an infinite or NaN sum here and
        # a NaN m2 (inf - inf); its sum is stored as NaN, as Inductor's
        # recurrence returns the mean.
        spread = block_sum - block_sum
        sum_p[tile_s, tile_c] = torch.where(torch.isnan(spread), spread, block_sum)
        # The block's mean rides along: the combine takes the block means'
        # deviations from it, without a division per block.
        mean_p[tile_s, tile_c] = pivot + offset
        m2_p[tile_s, tile_c] = shifted_sq - shifted_sum * offset
        weight_p[tile_s, tile_c] = hl.zeros(
            [tile_s, tile_c], dtype=torch.float32
        ) + count.to(torch.float32)
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
    # sum_p, mean_p, m2_p: [splits, C] fp64 partials, weight_p their fp32
    # counts (zero for a block past the end); running_*: [C] fp32, updated in
    # place.
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
        sum_p, mean_p, m2_p, weight_p = bn_welford_partials(
            x.view(n, c, h * w).permute(0, 2, 1), SPLITS
        )
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
