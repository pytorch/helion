"""Kernels shared by the tests of the dtype a vector-lane fold accumulates in
(``test_cute_fold_accumulator_dtype.py`` and its ``_cuda`` module): reductions
over a tiled dim, whose vector lanes ``hoist_warp_reduce`` folds into one
accumulator per thread, in dtypes other than fp32, with reduce inputs whose
definitions are not conversions (a load, a product) or convert to a narrower
dtype than the reduction's.  The reduce inputs that are negations are those
of ``integer_extremum_kernels``' row maxima and minima, which these tests
run too."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Layouts of the row kernels over 256 columns: 32 column threads of one warp
# with one or two packets of four columns each, and 64 threads on two warps,
# which combine through shared memory, with one packet of four columns or
# two packets of two.
ONE_WARP = {
    "block_sizes": [1, 128],
    "num_threads": [1, 32],
    "cute_vector_widths": [1, 4],
}
TWO_PACKETS = {**ONE_WARP, "block_sizes": [1, 256]}
TWO_WARPS = {**TWO_PACKETS, "num_threads": [1, 64]}
TWO_WARPS_TWO_PACKETS = {**TWO_WARPS, "cute_vector_widths": [1, 2]}
# The same threads with one column per packet: no vector lanes to fold.
SCALAR_LANES = {**ONE_WARP, "cute_vector_widths": [1, 1]}

# ``split_column_moments`` on 512-row tiles: 32 row threads beside 32 column
# threads (32 warps, combined through shared memory), four rows and eight
# columns per packet, two row packets per thread and tile.
SPLIT_MOMENTS = {
    "block_sizes": [4, 32, 512],
    "num_threads": [1, 32, 32],
    "cute_vector_widths": [2, 8, 4],
    "cute_lane_layouts": ["strided", "strided", "blocked"],
    "cute_lane_unroll": 2,
    "cute_defer_tile_reduce": True,
    "cute_tail_peel": True,
}
SPLIT_MOMENTS_SCALAR_LANES = {
    "block_sizes": [4, 32, 512],
    "num_threads": [1, 32, 32],
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_sums(x: torch.Tensor) -> torch.Tensor:
    """Row sums in ``x``'s dtype (float64, int64): each reduce input is a load."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_n in hl.tile(n):
            acc = acc + x[tile_m, tile_n].sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_sums_as_half(x: torch.Tensor) -> torch.Tensor:
    """Row sums of an fp32 tensor's elements rounded to fp16: each reduce
    input is a conversion narrower than the fp32 the reduction combines in."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float16)
        for tile_n in hl.tile(n):
            acc = acc + x[tile_m, tile_n].to(torch.float16).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_square_sums(x: torch.Tensor) -> torch.Tensor:
    """Row sums of squares in ``x``'s dtype: each reduce input is a product."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=x.dtype)
        for tile_n in hl.tile(n):
            value = x[tile_m, tile_n]
            acc = acc + (value * value).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_dots_int32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row dot products of int32 tensors, summed in int32."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.int32)
        for tile_n in hl.tile(n):
            acc = acc + torch.sum(
                x[tile_m, tile_n] * y[tile_m, tile_n], dim=-1, dtype=torch.int32
            )
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_dots_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row dot products of int32 tensors, summed in int64 as ``torch.sum``
    promotes them: each reduce input is the product's conversion to int64."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            acc = acc + (x[tile_m, tile_n] * y[tile_m, tile_n]).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_maxima(x: torch.Tensor, lowest: hl.constexpr) -> torch.Tensor:
    """Row maxima in ``x``'s dtype, starting from ``lowest``."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        acc = hl.full([tile_m], lowest, dtype=x.dtype)
        for tile_n in hl.tile(n):
            acc = torch.maximum(acc, x[tile_m, tile_n].amax(-1))
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_minima(x: torch.Tensor, highest: hl.constexpr) -> torch.Tensor:
    """Row minima in ``x``'s dtype, starting from ``highest``."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        acc = hl.full([tile_m], highest, dtype=x.dtype)
        for tile_n in hl.tile(n):
            acc = torch.minimum(acc, x[tile_m, tile_n].amin(-1))
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_logsumexp(x: torch.Tensor, dtype: hl.constexpr) -> torch.Tensor:
    """The online log-sum-exp of the rows of a half-precision ``x``, its
    running maximum and sum in ``dtype``.  In ``x``'s dtype every exponential
    is a half-precision value: the sum's input is a half conversion."""
    m, n = x.size()
    # pyrefly: ignore [no-matching-overload]
    out = torch.empty([m], dtype=dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        running_max = hl.full([tile_m], float("-inf"), dtype=dtype)
        # pyrefly: ignore [bad-argument-type]
        running_sum = hl.zeros([tile_m], dtype=dtype)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n]
            new_max = torch.maximum(running_max, torch.amax(values, dim=1))
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                values - new_max[:, None]
            ).sum(dim=1)
            running_max = new_max
        out[tile_m] = running_max + torch.log(running_sum)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_logsumexp_of_doubled(x: torch.Tensor, dtype: hl.constexpr) -> torch.Tensor:
    """``row_logsumexp`` of ``2 * x``: the maximum's input is the product's
    conversion to ``x``'s half-precision dtype, which the exponent reads."""
    m, n = x.size()
    # pyrefly: ignore [no-matching-overload]
    out = torch.empty([m], dtype=dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        running_max = hl.full([tile_m], float("-inf"), dtype=dtype)
        # pyrefly: ignore [bad-argument-type]
        running_sum = hl.zeros([tile_m], dtype=dtype)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n] * 2
            new_max = torch.maximum(running_max, torch.amax(values, dim=1))
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                values - new_max[:, None]
            ).sum(dim=1)
            running_max = new_max
        out[tile_m] = running_max + torch.log(running_sum)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def split_column_moments(
    x: torch.Tensor, rows_per_split: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each block of ``rows_per_split`` rows (the last one partial), the
    column sums of ``x - 1`` and of its square in float64, walked by an
    explicit row loop with masked loads: the shifted moments of
    better-benchmark's BatchNorm partials, at test sizes."""
    rows, n = x.size()
    # pyrefly: ignore [unsupported-operation]
    splits = (rows + rows_per_split - 1) // rows_per_split
    total = torch.empty([splits, n], dtype=torch.float64, device=x.device)
    squares = torch.empty([splits, n], dtype=torch.float64, device=x.device)
    for tile_s, tile_n in hl.tile([splits, n]):
        acc = hl.zeros([tile_s, tile_n], dtype=torch.float64)
        acc_sq = hl.zeros([tile_s, tile_n], dtype=torch.float64)
        # pyrefly: ignore [no-matching-overload]
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            mask = valid[:, :, None]
            value = hl.load(x, [row, tile_n], extra_mask=mask).to(torch.float64)
            shifted = torch.where(mask, value - 1.0, 0.0)
            acc = acc + torch.sum(shifted, dim=1)
            acc_sq = acc_sq + torch.sum(shifted * shifted, dim=1)
        total[tile_s, tile_n] = acc
        squares[tile_s, tile_n] = acc_sq
    return total, squares


def split_column_moments_reference(
    x: torch.Tensor, rows_per_split: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``split_column_moments`` with torch."""
    rows, n = x.shape
    splits = -(-rows // rows_per_split)
    shifted = torch.zeros(
        splits * rows_per_split, n, dtype=torch.float64, device=x.device
    )
    shifted[:rows] = x.double() - 1.0
    shifted = shifted.view(splits, rows_per_split, n)
    return shifted.sum(1), (shifted * shifted).sum(1)
