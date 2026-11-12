"""Kernels that read a padded ``hl.arange`` of a row's extent beside the row.

Shared by ``test/test_rolled_reduction_positions.py`` (their numerics and
reduction dims) and ``test/test_config_api.py`` (the ``reduction_loops``
slots of the configs older versions saved for them).  For a host int ``n``
that is not a power of two, ``hl.arange(n)`` is emitted over
``next_power_of_2(n)`` lanes; its subscript shares the row's reduction dim
of ``n``, which then stays persistent.
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
