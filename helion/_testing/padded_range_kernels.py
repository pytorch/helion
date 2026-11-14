"""Kernels that read a padded ``hl.arange`` of a row's extent beside the
row, and grid kernels that reduce a load over one.

For a host int ``n`` that is not a power of two, ``hl.arange(n)`` is emitted
over ``next_power_of_2(n)`` lanes.  Beside a row, its subscript shares the
row's reduction dim of ``n``, which then stays persistent; a grid kernel's
load over it has the padded size, and so has the reduction dim of its sum.

The row kernels are shared by ``test/test_rolled_reduction_positions.py``
(their numerics and reduction dims) and ``test/test_config_api.py`` (the
``reduction_loops`` slots of the configs older versions saved for them), the
grid kernels by ``test/test_padded_range_reduction.py`` (their codegen) and
``test/test_reductions.py`` (their numerics).
"""

from __future__ import annotations

import torch

import helion.language as hl


def host_arange_mark(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Each row with the column at ``pos`` set to -1: the range is read
    pointwise beside the row, inside the row's loop."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        cols = hl.arange(n)
        out[tile_m, :] = torch.where(cols[None, :] == pos[tile_m][:, None], -1.0, row)
    return out


def host_arange_mark_then_row_sums(
    x: torch.Tensor, pos: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``host_arange_mark``, then the row sums of ``y`` and ``z``, whose dims
    take the ``reduction_loops`` slots after the marked row's."""
    m, n = x.shape
    out = torch.empty_like(x)
    y_sums = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)
    z_sums = torch.empty([z.size(0)], dtype=z.dtype, device=z.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        cols = hl.arange(n)
        out[tile_m, :] = torch.where(cols[None, :] == pos[tile_m][:, None], -1.0, row)
    for tile_y in hl.tile(y.size(0)):
        y_sums[tile_y] = y[tile_y, :].sum(-1)
    for tile_z in hl.tile(z.size(0)):
        z_sums[tile_z] = z[tile_z, :].sum(-1)
    return out, y_sums, z_sums


def grid_pick_column(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Each row's element at ``pos``, summed out of a load over the range
    that is typed before the range's subscript names the extent."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        cols = hl.arange(n)
        # Typed before the range's subscript names the extent of 200.
        values = hl.load(x, [i, cols])
        out[i] = torch.where(cols[None, :] == pos[i], values[None, :], 0.0).sum(-1)
    return out


def grid_padded_range_sum(x: torch.Tensor) -> torch.Tensor:
    """Sums each row through a load over ``hl.arange(n)``, which pads ``n``."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        out[i] = hl.load(x, [i, hl.arange(n)]).sum(-1)
    return out


def grid_offset_range_sum(x: torch.Tensor) -> torch.Tensor:
    """Sums 200 columns of each row from the fourth, through a padded range."""
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for i in hl.grid(x.size(0)):
        out[i] = hl.load(x, [i, hl.arange(3, 203)]).sum(-1)
    return out


def grid_padded_range_sum_and_max(x: torch.Tensor) -> torch.Tensor:
    """Reduces ``2 * x + 1`` over a padded range, one in its padding lanes."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        v = hl.load(x, [i, hl.arange(n)]) * 2.0 + 1.0
        out[i] = v.sum(-1) + v.amax(-1)
    return out


def grid_padded_range_sum_scales_a_row(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Scales each row of ``y``, in an inner loop, by the sum of the row of
    ``x`` through a load over ``hl.arange(n)``, which pads ``n``."""
    out = torch.empty_like(y)
    for i in hl.grid(x.size(0)):
        s = hl.load(x, [i, hl.arange(x.size(1))]).sum(-1)
        for j in hl.grid(y.size(1)):
            out[i, j] = s * y[i, j]
    return out


def grid_padded_range_sums_accumulated(x: torch.Tensor) -> torch.Tensor:
    """Adds up the sums of ``x[i, j, :]`` over ``j`` in an inner loop, each
    through a load over ``hl.arange(n)``, which pads ``n``."""
    m, k, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        acc = hl.zeros([], dtype=x.dtype)
        for j in hl.grid(k):
            acc = acc + hl.load(x, [i, j, hl.arange(n)]).sum(-1)
        out[i] = acc
    return out


def grid_store_through_a_view_of_the_range(x: torch.Tensor) -> torch.Tensor:
    """Adds each row's sum to its columns 1..5, loaded and stored through
    ``hl.arange(1, 6).reshape(5)``, a view of the padded range in its own
    shape."""
    out = torch.zeros_like(x)
    for i in hl.grid(x.size(0)):
        s = x[i, :].sum()
        cols = hl.arange(1, 6).reshape(5)
        hl.store(out, [i, cols], hl.load(x, [i, cols]) + s)
    return out
