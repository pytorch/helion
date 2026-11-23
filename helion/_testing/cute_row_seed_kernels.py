"""Row kernels at the edges of the CuTe row heuristics' gate
(``_row_tile_axes``): kernels that reduce a second block beside the row of
their reduction loop, and kernels with a block size that is no tile axis of
the rows.

Plain functions, which ``test/test_cute_row_seed_gates.py`` and
``test/test_cute_row_packet_seeds.py`` bind with the settings they need.
"""

from __future__ import annotations

import torch

import helion.language as hl


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """A row's max and sum: two reductions of one block."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = torch.softmax(row, dim=-1).to(out.dtype)
    return out


def softmax_rows_over_two_tiles(x: torch.Tensor) -> torch.Tensor:
    """``softmax_rows`` over ``(batch, row)`` tiles: two block sizes, both
    tile axes of the rows."""
    out = torch.empty_like(x)
    for tile_b, tile_m in hl.tile([x.size(0), x.size(1)]):
        row = x[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = torch.softmax(row, dim=-1).to(out.dtype)
    return out


def row_and_half_norms(x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm over the row and one over its first half: the kernel
    reduces two blocks, the row and its half."""
    rows, dim = x.shape
    half = dim // 2
    out = torch.empty_like(x)
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        lo = x[tile, :half].to(torch.float32)
        lstd = torch.rsqrt(torch.mean(lo * lo, dim=-1, keepdim=True) + 1e-6)
        out[tile, :half] = (lo * rstd * lstd).to(out.dtype)
        out[tile, half:] = (x[tile, half:].to(torch.float32) * rstd).to(out.dtype)
    return out


def half_then_row_norms(x: torch.Tensor) -> torch.Tensor:
    """``row_and_half_norms`` with the half reduced first, so the half row
    holds the kernel's reduction loop and the whole row is the second
    reduced block."""
    half = x.size(1) // 2
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        lo = x[tile, :half].to(torch.float32)
        lstd = torch.rsqrt(torch.mean(lo * lo, dim=-1, keepdim=True) + 1e-6)
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile, :] = (row * rstd * lstd).to(out.dtype)
    return out


def row_and_half_norms_over_two_tiles(x: torch.Tensor) -> torch.Tensor:
    """``row_and_half_norms`` over ``(token, head)`` tiles."""
    half = x.size(2) // 2
    out = torch.empty_like(x)
    for tile_t, tile_h in hl.tile([x.size(0), x.size(1)]):
        row = x[tile_t, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        lo = x[tile_t, tile_h, :half].to(torch.float32)
        lstd = torch.rsqrt(torch.mean(lo * lo, dim=-1, keepdim=True) + 1e-6)
        out[tile_t, tile_h, :half] = (lo * rstd * lstd).to(out.dtype)
        out[tile_t, tile_h, half:] = (
            x[tile_t, tile_h, half:].to(torch.float32) * rstd
        ).to(out.dtype)
    return out


def rows_over_two_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each row of ``x`` divided by its sum plus the sum of the row of ``y``,
    a tensor of another width: the kernel reduces two blocks."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        total = row.sum(-1, keepdim=True)
        other = y[tile, :].to(torch.float32).sum(-1, keepdim=True)
        out[tile, :] = (row / (total + other)).to(out.dtype)
    return out


def row_norm_then_tiled_columns(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization, then a device loop that tiles the columns of
    ``y``: a block size that is no axis of the root grid."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
        for tile_c in hl.tile(y.size(1)):
            doubled[tile, tile_c] = y[tile, tile_c] * 2
    return out, doubled


def grid_row_norm_then_tiled_columns(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``row_norm_then_tiled_columns`` with the rows on ``hl.grid``: the
    kernel's one block size tiles the columns of ``y``."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    for i in hl.grid(x.size(0)):
        row = x[i, :].to(torch.float32)
        out[i, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
        for tile_c in hl.tile(y.size(1)):
            doubled[i, tile_c] = y[i, tile_c] * 2
    return out, doubled


def softmax_rows_inner_block_size_first(x: torch.Tensor) -> torch.Tensor:
    """``softmax_rows_over_two_tiles`` whose inner tile axis takes a block
    size registered before the loop: the block sizes' order is not the tile
    axes' order."""
    out = torch.empty_like(x)
    block_size = hl.register_block_size(x.size(1))
    for tile_b, tile_m in hl.tile(
        [x.size(0), x.size(1)], block_size=[None, block_size]
    ):
        row = x[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = torch.softmax(row, dim=-1).to(out.dtype)
    return out


def row_norm_beside_a_loop_sharing_a_block_size(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization over ``(batch, row)`` tiles, then a root loop
    that tiles the vector ``y``, of another length, with the row axis' block
    size: a block size that two loops use over different extents."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    block_size = hl.register_block_size(x.size(1))
    for tile_b, tile_m in hl.tile(
        [x.size(0), x.size(1)], block_size=[None, block_size]
    ):
        row = x[tile_b, tile_m, :].to(torch.float32)
        out[tile_b, tile_m, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for tile in hl.tile(y.size(0), block_size=block_size):
        doubled[tile] = y[tile] * 2
    return out, doubled


def first_elements_then_row_norm(x: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """A root loop of its own over the rows, tiled, then the row
    normalization in a second root loop: the first loop's block size is no
    tile axis of the reduced rows."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        first[tile] = x[tile, 0].to(torch.float32)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    return out


def grid_first_elements_then_row_norm(
    x: torch.Tensor, first: torch.Tensor
) -> torch.Tensor:
    """``first_elements_then_row_norm`` with the first loop on ``hl.grid``:
    the kernel's one block size tiles the reduced rows."""
    out = torch.empty_like(x)
    for i in hl.grid(x.size(0)):
        first[i] = x[i, 0].to(torch.float32)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    return out


def tiled_first_elements_then_grid_row_norm(
    x: torch.Tensor, first: torch.Tensor
) -> torch.Tensor:
    """A tiled root loop without a reduction, then the row normalization on
    ``hl.grid``: the kernel's one block size is an axis of a root grid, but
    not of the one that reduces the rows."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        first[tile] = x[tile, 0].to(torch.float32)
    for i in hl.grid(x.size(0)):
        row = x[i, :].to(torch.float32)
        out[i, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    return out


def tiled_first_elements_then_grid_doubled_rows(
    x: torch.Tensor, first: torch.Tensor
) -> torch.Tensor:
    """``tiled_first_elements_then_grid_row_norm`` without the reduction:
    the row loop rolls a full slice that nothing reduces."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        first[tile] = x[tile, 0].to(torch.float32)
    for i in hl.grid(x.size(0)):
        out[i, :] = (x[i, :].to(torch.float32) * 2).to(out.dtype)
    return out


def row_norm_then_grid_doubled_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization, then an ``hl.grid`` root loop that sweeps rows of
    the same width: the second loop shares the row's block and has no block
    size."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for i in hl.grid(x.size(0)):
        doubled[i, :] = x[i, :] * 2
    return out, doubled


def two_row_norms_sharing_a_block_size(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two root loops over the rows with one registered block size, each
    reducing the row."""
    out = torch.empty_like(x)
    scaled = torch.empty_like(x)
    block_size = hl.register_block_size(x.size(0))
    for tile in hl.tile(x.size(0), block_size=block_size):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for tile in hl.tile(x.size(0), block_size=block_size):
        row = x[tile, :].to(torch.float32)
        scaled[tile, :] = (row / row.amax(-1, keepdim=True)).to(scaled.dtype)
    return out, scaled


def two_row_norms_in_swapped_tile_orders(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two root loops that each reduce rows of one width, over the same two
    registered block sizes in opposite tile orders: ``x`` is ``[batch, head,
    row]`` and ``y`` is ``[head, batch, row]``."""
    out = torch.empty_like(x)
    scaled = torch.empty_like(y)
    batch = hl.register_block_size(x.size(0))
    heads = hl.register_block_size(x.size(1))
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[batch, heads]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        out[tile_b, tile_h, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for tile_h, tile_b in hl.tile([y.size(0), y.size(1)], block_size=[heads, batch]):
        row = y[tile_h, tile_b, :].to(torch.float32)
        scaled[tile_h, tile_b, :] = (row / row.amax(-1, keepdim=True)).to(scaled.dtype)
    return out, scaled


def row_norm_over_two_tiles_then_doubled_rows(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization over ``(batch, head)`` tiles, then a root loop
    that sweeps the rows of ``y``, of the same width, over the batch axis'
    block size alone."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    batch = hl.register_block_size(x.size(0))
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[batch, None]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        out[tile_b, tile_h, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for tile in hl.tile(y.size(0), block_size=batch):
        doubled[tile, :] = y[tile, :] * 2
    return out, doubled


def row_norm_then_grid_row_norm(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization over tiles, then one on ``hl.grid`` over the same
    rows: both root loops reduce the row's block, and only the first has the
    kernel's block size."""
    out = torch.empty_like(x)
    scaled = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for i in hl.grid(x.size(0)):
        row = x[i, :].to(torch.float32)
        scaled[i, :] = (row / row.amax(-1, keepdim=True)).to(scaled.dtype)
    return out, scaled


def two_row_norms_over_tiles_of_their_own(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A normalization of the rows of ``q`` and one of the rows of ``k``, of
    the same width, in two root loops: each loop reduces the row's block
    over a block size of its own."""
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    for tile in hl.tile(q.size(0)):
        row = q[tile, :].to(torch.float32)
        q_out[tile, :] = (row / row.sum(-1, keepdim=True)).to(q_out.dtype)
    for tile in hl.tile(k.size(0)):
        row = k[tile, :].to(torch.float32)
        k_out[tile, :] = (row / row.amax(-1, keepdim=True)).to(k_out.dtype)
    return q_out, k_out


def row_norm_beside_doubled_rows(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization of ``x`` and, in the same loop, the rows of
    ``y``, of another width, doubled: a full slice swept beside the one
    reduced row."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
        doubled[tile, :] = y[tile, :] * 2
    return out, doubled


def row_norm_then_tiled_doubled_rows(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization, then a second tiled root loop that sweeps the
    same rows without reducing them: its block size tiles rows that nothing
    reduces."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    for tile in hl.tile(x.size(0)):
        doubled[tile, :] = x[tile, :] * 2
    return out, doubled
