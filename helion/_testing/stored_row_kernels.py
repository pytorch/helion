"""Kernels and inputs shared by the tests of a tile loop that stores its
tile per element (``cute_store_packet_tiers``): rows doubled, and rows whose
view starts at a chosen element of padded storage."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def planted_rows(
    rows: int,
    columns: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """``[rows, columns]`` random values, a view of rows padded to 1024 (so
    the view's rows start at one 16-byte phase; the padding holds 1e4); row
    0 ends in 50.0, row 1 holds 50.0 in its third column from the end, and
    row 2 is all ``-inf``."""
    generator = torch.Generator().manual_seed(columns)
    base = torch.full((rows + 1, 1024), 1e4)
    view = base[:rows, :columns]
    view.copy_(torch.randn(rows, columns, generator=generator).mul_(4))
    view[0, columns - 1] = 50.0
    view[1, columns - 3] = 50.0
    view[2] = float("-inf")
    return base.to(device=device, dtype=dtype)[:rows, :columns]


def row_view(wide: torch.Tensor, view: str, columns: int) -> torch.Tensor:
    """``columns`` elements of every row of ``wide`` as an output view:
    from element ``int(view)`` of the row on, or every other column for
    ``"strided"`` (the lanes are then not adjacent elements)."""
    if view == "strided":
        return wide[:, : 2 * columns : 2]
    offset = int(view)
    return wide[:, offset : offset + columns]


@helion.kernel(backend="cute", static_shapes=True)
def doubled_rows_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out = 2 * x`` row by row into the caller's tensor, whose base and
    strides are the caller's: a view that starts off a packet boundary in
    its storage, or whose columns are not adjacent elements."""
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def doubled_rows_into_two_dtypes(
    x: torch.Tensor, half: torch.Tensor, full: torch.Tensor
) -> torch.Tensor:
    """``half = 2 * x`` (bf16) and ``full = x + 1`` (fp32) row by row into
    the caller's tensors from one device loop: stores of two element sizes,
    whose packets halve along chains of different lengths."""
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n]
            half[tile_m, tile_n] = (v * 2.0).to(torch.bfloat16)
            full[tile_m, tile_n] = v + 1.0
    return half


@helion.kernel(backend="cute", static_shapes=True)
def doubled_rows_from(x: torch.Tensor, start: int) -> torch.Tensor:
    """``out = 2 * x`` from column ``start`` on, zero ahead of it: the
    tiles begin where the caller says, not a multiple of their lanes into
    the row."""
    m, n = x.shape
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(start, n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def row_fill_from(scale: torch.Tensor, out: torch.Tensor, start: int) -> torch.Tensor:
    """``out[row, start:] = 2 * scale[row]``: the tiles begin at the caller's
    column and the stored value is loaded per row, not along the tile, so no
    packet load's guard stands beside the store's: a fragment whose first
    lane is off the whole packet's boundary runs a narrower tier (a loop
    that loads along the same columns runs its scalar loop there, because
    every arm keeps the loads' whole packets)."""
    m, n = out.shape
    for tile_m in hl.tile(m):
        row = scale[tile_m] * 2.0
        for tile_n in hl.tile(start, n):
            out[tile_m, tile_n] = row[:, None] + hl.zeros(
                [tile_m, tile_n], dtype=out.dtype
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, cute_rng_stream="philox4")
def random_rows_into(out: torch.Tensor, seed: int) -> torch.Tensor:
    """``out = hl.rand`` row by row into the caller's fp32 tensor.  With
    ``cute_rng_packet`` a thread draws four values per Philox call, and the
    Philox lowering has the affine vectorizer pack their stores itself: the
    lane loops' pass declines a loop with a random call.  No packet load
    stands beside the store, but the packet has a guard of its own: its
    first value's position in the stream, ``row * columns + column``, is a
    multiple of four.  In a contiguous output that is the store's alignment
    too; a view of wider rows (``columns`` a multiple of four, the rows a
    different number of elements apart) passes the packet's guard on rows
    whose stores are off the 16-byte boundary."""
    m, n = out.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = hl.rand([tile_m, tile_n], seed=seed)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def doubled_rows_past_three(x: torch.Tensor) -> torch.Tensor:
    """``out = 2 * x`` from column 3 on, zero ahead of it: a loop of static
    extent whose tiles begin off every packet boundary of their row."""
    m, n = x.shape
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(3, n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def doubled_pairs(x: torch.Tensor) -> torch.Tensor:
    """``out[..., 0] = 2 * x`` and ``out[..., 1] = x + 1``: the device loop's
    tile is ahead of a component index, not on the stored tensor's last
    dimension."""
    m, n = x.shape
    out = torch.empty([m, n, 2], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n]
            out[tile_m, tile_n, 0] = v * 2.0
            out[tile_m, tile_n, 1] = v + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def centered_slices(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row less its sum, stored through a full slice under a device
    loop over the rows that carries a column partial (the nest of a
    LayerNorm backward): the loop's tile is not on the stored tensor's last
    dimension either."""
    splits, rows, hidden = x.shape
    out = torch.empty_like(x)
    partial = torch.empty([splits, hidden], dtype=torch.float32, device=x.device)
    for tile_s in hl.tile(splits):
        acc = hl.zeros([tile_s, hidden], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            row = x[tile_s, tile_r, :]
            out[tile_s, tile_r, :] = row - torch.sum(row, dim=-1, keepdim=True)
            acc = acc + torch.sum(row, dim=1)
        partial[tile_s, :] = acc
    return out, partial


@helion.kernel(backend="cute", static_shapes=True)
def doubled_rows(x: torch.Tensor) -> torch.Tensor:
    """``out = 2 * x`` row by row: a device loop that only stores along its
    tile, into rows whose alignment follows from the row length."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return out
