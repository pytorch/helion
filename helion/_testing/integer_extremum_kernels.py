"""Kernels shared by the tests of integer maxima and minima inside CuTe
reductions (``test_cute_integer_extrema.py`` and its ``_cuda`` module): the
maximum or minimum of negated integers per row, over column tiles, over the
whole row, and over column tiles ahead of a second column loop.  Each reduce
input is a negated load: ptxas (CUDA 13.3) loses such a negation when it
fuses a chain of int32 or int16 maxima or minima, which is what a thread's
accumulate over its elements is.  An argmin over column tiles has such a
chain of indices, which are never negations.  The last kernels have a chain
without a reduction: the maximum and the minimum of three negated rows of a
register region.  ``adversarial_rows`` builds the input of the row kernels.
The tests of the vector-lane fold (``test_cute_fold_accumulator_dtype.py``
and its ``_cuda`` module) run ``row_maxima_of_negated`` and
``row_minima_of_negated`` on those rows too."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Layouts of the tiled row kernels over 256 columns, one column per load, so
# that each thread accumulates its columns one element at a time: four
# columns on each of the 32 threads of one warp; four on each of 64 threads,
# whose two warps combine through shared memory; and sixteen on one thread.
ONE_WARP = {"block_sizes": [1, 128], "num_threads": [1, 32]}
TWO_WARPS = {"block_sizes": [1, 256], "num_threads": [1, 64]}
ONE_THREAD = {"block_sizes": [1, 16], "num_threads": [1, 1]}
# The whole-row kernels with a reduction loop of 64 columns per trip: each
# thread accumulates four trips.
LOOPED = {"block_sizes": [1], "reduction_loops": [64]}
# ``rows_minus_maxima_of_negated`` with each of its column loops as one tile
# of 256 columns on 32 threads, four rows per block, and its reductions
# ordered over the whole tile (``cute_reduction_sequence``): each thread
# accumulates eight columns in that lowering's own accumulate.
SEQUENCE = {
    "block_sizes": [4, 256, 256],
    "num_threads": [4, 32, 32],
    "cute_reduction_sequence": "resident",
}
# The same kernel on 128 threads (four warps, two columns per thread): the
# warps' partials combine through shared memory.
SEQUENCE_FOUR_WARPS = {
    "block_sizes": [1, 256, 256],
    "num_threads": [1, 128, 128],
    "cute_reduction_sequence": "resident",
}
# ``whole_row_argmin_in_nested_tiles`` over (4, 16, 1024): its 1024 columns
# are split over the lanes of the threads that reduce them.
NESTED = {"block_sizes": [4, 16]}


def adversarial_rows(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sixteen rows of 256 ordinary and extreme values of an integer
    ``dtype``: random ones over its range and small ones, its bounds alone,
    mixed and with small offsets, and one outlier among equal values in each
    of the four columns a thread takes from a tile."""
    info = torch.iinfo(dtype)
    low, high = info.min, info.max
    generator = torch.Generator().manual_seed(0)
    column = torch.arange(256)
    x = torch.randint(low, high, (16, 256), generator=generator)
    x[1] = torch.randint(max(low, -100), 100, (256,), generator=generator)
    x[2] = high - column % 7
    x[3] = low + column % 7
    x[4] = torch.where(column % 2 == 0, high, low)
    x[5] = torch.where(column % 4 == 1, high - 1, low + 1)
    x[6] = high
    x[7] = low
    for lane in range(4):
        x[8 + lane] = 0
        x[8 + lane, 132 + lane] = 1
        x[12 + lane] = high
        x[12 + lane, 4 + lane] = 5
    return x.to(dtype).to(device)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_maxima_of_negated(x: torch.Tensor, lowest: hl.constexpr) -> torch.Tensor:
    """Row maxima of ``-x`` in ``x``'s integer dtype, over column tiles."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        acc = hl.full([tile_m], lowest, dtype=x.dtype)
        for tile_n in hl.tile(n):
            acc = torch.maximum(acc, torch.amax(-x[tile_m, tile_n], dim=-1))
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_minima_of_negated(x: torch.Tensor, highest: hl.constexpr) -> torch.Tensor:
    """Row minima of ``-x`` in ``x``'s integer dtype, over column tiles."""
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        acc = hl.full([tile_m], highest, dtype=x.dtype)
        for tile_n in hl.tile(n):
            acc = torch.minimum(acc, torch.amin(-x[tile_m, tile_n], dim=-1))
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def whole_row_maxima_of_negated(x: torch.Tensor) -> torch.Tensor:
    """Row maxima of ``-x``, each row reduced whole."""
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(-x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def whole_row_minima_of_negated(x: torch.Tensor) -> torch.Tensor:
    """Row minima of ``-x``, each row reduced whole."""
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(-x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def rows_minus_maxima_of_negated(x: torch.Tensor) -> torch.Tensor:
    """``x`` minus its row's maximum of ``-x``, in fp32: the integer maximum
    of each column tile is carried and consumed in fp32, by a second column
    loop (two sibling loops, which the ordered lowering of a tile's
    reductions takes; it declines an integer carry)."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        top = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n):
            top = torch.maximum(
                top, torch.amax(-x[tile_m, tile_n], dim=-1).to(torch.float32)
            )
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n].to(torch.float32) - top[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_argmin_over_tiles(x: torch.Tensor) -> torch.Tensor:
    """The position of each row's minimum, over column tiles: the candidate
    indices of each tile's argmin reduce through an integer minimum."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        best = hl.full([tile_m], float("inf"), dtype=torch.float32)
        where = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n]
            low = values.amin(-1)
            index = values.argmin(-1) + tile_n.begin
            where = torch.where(low < best, index, where)
            best = torch.minimum(best, low)
        out[tile_m] = where
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def whole_row_argmin_in_nested_tiles(x: torch.Tensor) -> torch.Tensor:
    """The position of the minimum of each row of a 3-D ``x``, each row
    reduced whole inside two nested tile loops: the other emitter of an
    argmin's candidate indices."""
    b, m, _ = x.size()
    out = torch.empty([b, m], dtype=torch.int64, device=x.device)
    for tile_b in hl.tile(b):
        for tile_m in hl.tile(m):
            out[tile_b, tile_m] = torch.argmin(x[tile_b, tile_m, :], dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_argmin_of_negated(x: torch.Tensor, highest: hl.constexpr) -> torch.Tensor:
    """The position of each row's minimum of ``-x`` in ``x``'s integer dtype,
    over column tiles: the lanes accumulate the minimum of the negated
    values, a chain like any other, and then its candidate indices."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_m], highest, dtype=x.dtype)
        where = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            values = -x[tile_m, tile_n]
            low = values.amin(-1)
            index = values.argmin(-1) + tile_n.begin
            where = torch.where(low < best, index, where)
            best = torch.minimum(best, low)
        out[tile_m] = where
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def register_maxima_of_negated(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Per row of three ``[rows, groups, width]`` tensors, the maximum of
    ``-a``, ``-b`` with its groups swapped in pairs, and ``-c``: the gather
    makes the row a register region, whose program holds the two maxima."""
    out = torch.empty_like(a)
    for row in hl.grid(a.size(0)):
        groups = hl.arange(a.size(1))[:, None]
        peer = torch.gather(b[row, :, :], 0, (groups ^ 1).expand(-1, a.size(2)))
        out[row, :, :] = torch.maximum(
            torch.maximum(-a[row, :, :], -peer), -c[row, :, :]
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def register_minima_of_negated(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """The minimum of the same three negated rows of a register region."""
    out = torch.empty_like(a)
    for row in hl.grid(a.size(0)):
        groups = hl.arange(a.size(1))[:, None]
        peer = torch.gather(b[row, :, :], 0, (groups ^ 1).expand(-1, a.size(2)))
        out[row, :, :] = torch.minimum(
            torch.minimum(-a[row, :, :], -peer), -c[row, :, :]
        )
    return out
