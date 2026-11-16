"""Reverse scans whose lane loop also wraps a device loop reducing over the scan.

Shared by the CuTe scan codegen tests (``test/test_cute_scan_lowering.py``)
and their GPU twins (``test/test_cute_scan_numerics.py``).  Under a config
where one thread walks the scanned row, the scan's descending lane loop
also wraps the device loop, whose reduction over the same lanes
``interchange_lane_outside_serial_reductions`` reruns in a second nest.
"""

from __future__ import annotations

import torch

import helion.language as hl


def reverse_scan_then_row_sums(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    totals = x.new_empty([x.size(0), y.size(1)])
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.cumsum(x[i, :], dim=1, reverse=True)
        for tile_k in hl.tile(y.size(1)):
            totals[i, tile_k] = y[i, tile_k, :].sum(-1)
    return out, totals


def reverse_scan_then_column_sums(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros([x.size(1)], dtype=torch.float32, device=x.device)
    suffix = torch.empty_like(w)
    for tile_b in hl.tile(x.size(0)):
        suffix[tile_b] = hl.cumsum(w[tile_b], dim=0, reverse=True)
        for tile_m in hl.tile(x.size(1)):
            hl.atomic_add(sums, [tile_m], x[tile_b, tile_m].sum(dim=0))
    return sums, suffix


def reverse_scan_then_centering(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centers each ``y[i, k, :]`` in place."""
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.cumsum(x[i, :], dim=1, reverse=True)
        for tile_k in hl.tile(y.size(1)):
            row = y[i, tile_k, :]
            y[i, tile_k, :] = row - row.sum(-1, keepdim=True)
    return out, y


def reverse_scan_then_centering_with_totals(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centers each ``y[i, k, :]`` in place and adds up, per column, the
    rows the loop over ``k`` read: the first nest keeps that load of ``y``
    ahead of the update, each iteration reading only its own row."""
    out = torch.empty_like(x)
    totals = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.cumsum(x[i, :], dim=1, reverse=True)
        acc = hl.zeros([i, y.size(2)], dtype=torch.float32)
        for k in hl.grid(y.size(1)):
            row = y[i, k, :]
            acc = acc + row
            y[i, k, :] = row - row.sum(-1, keepdim=True)
        totals[i, :] = acc
    return out, y, totals
