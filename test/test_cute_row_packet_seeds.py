"""Row seeds outside ``CuteReductionTileHeuristic``
(``heuristics/cute_row_packets.py``): rows that are no whole number of warps
of 16-byte packets, which pack several rows into a warp; the warp rows of
grids too large for one row per CTA; and the slices a kernel sweeps beside
its row reduction, which take one packet per row thread."""

from __future__ import annotations

import functools
from itertools import starmap
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics import cute_row_packets
from helion._compiler.cute.autotune.heuristics.cute import CuteReductionTileHeuristic
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    CuteFlooredWarpRowHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    CuteLaneSplitRowHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    CuteMaskedTailRowHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    CuteSliceSweepPacketHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    CuteSubwarpRowHeuristic,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import _full_slices_meet
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    _row_is_no_run_of_packets,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    row_packet_seed_configs,
)
from helion._compiler.cute.autotune.heuristics.cute_row_packets import (
    slice_packet_entries,
)
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_row_seed_kernels import grid_first_elements_then_row_norm
from helion._testing.cute_row_seed_kernels import masked_softmax_backward_rows
from helion._testing.cute_row_seed_kernels import row_and_half_norms
from helion._testing.cute_row_seed_kernels import row_norm_then_tiled_columns
from helion._testing.cute_row_seed_kernels import rows_over_two_sums
from helion._testing.cute_row_seed_kernels import softmax_rows
from helion._testing.cute_row_seed_kernels import softmax_rows_inner_block_size_first
from helion._testing.cute_row_seed_kernels import softmax_rows_over_two_tiles
from helion._testing.registered_reduction import registered_scale_sum
import helion.language as hl
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

ROW_HEURISTICS = (
    CuteFlooredWarpRowHeuristic,
    CuteSubwarpRowHeuristic,
    CuteMaskedTailRowHeuristic,
    CuteSliceSweepPacketHeuristic,
)


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with cpu_target():
        yield


def residual_layernorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    rows, hidden = x.shape
    out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        summed = x[tile, :].to(torch.float32) + residual[tile, :].to(torch.float32)
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-6)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        out[tile, :] = y.to(out.dtype)
    return out


def rmsnorm_rotate(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """An RMSNorm over each ``(token, head)`` row, then a rotation that
    sweeps the row's two halves: the reduced row beside a half-row slice."""
    tokens, heads, dim = x.shape
    half = dim // 2
    out = torch.empty([tokens, heads, dim], dtype=x.dtype, device=x.device)
    for tile_t, tile_h in hl.tile([tokens, heads]):
        row = x[tile_t, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        lo = x[tile_t, tile_h, :half].to(torch.float32) * rstd * weight[:half]
        hi = x[tile_t, tile_h, half:].to(torch.float32) * rstd * weight[half:]
        cos_lo = cos[tile_t, :half].to(torch.float32)[:, None, :]
        cos_hi = cos[tile_t, half:].to(torch.float32)[:, None, :]
        sin_lo = sin[tile_t, :half].to(torch.float32)[:, None, :]
        sin_hi = sin[tile_t, half:].to(torch.float32)[:, None, :]
        out[tile_t, tile_h, :half] = (lo * cos_lo - hi * sin_lo).to(out.dtype)
        out[tile_t, tile_h, half:] = (hi * cos_hi + lo * sin_hi).to(out.dtype)
    return out


def flag_then_rmsnorm_rotate(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    flag: torch.Tensor,
) -> torch.Tensor:
    """``rmsnorm_rotate`` after a root loop of its own over the tokens."""
    tokens, heads, dim = x.shape
    half = dim // 2
    out = torch.empty([tokens, heads, dim], dtype=x.dtype, device=x.device)
    for tile in hl.tile(tokens):
        flag[tile] = cos[tile, 0]
    for tile_t, tile_h in hl.tile([tokens, heads]):
        row = x[tile_t, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        lo = x[tile_t, tile_h, :half].to(torch.float32) * rstd * weight[:half]
        hi = x[tile_t, tile_h, half:].to(torch.float32) * rstd * weight[half:]
        cos_lo = cos[tile_t, :half].to(torch.float32)[:, None, :]
        cos_hi = cos[tile_t, half:].to(torch.float32)[:, None, :]
        sin_lo = sin[tile_t, :half].to(torch.float32)[:, None, :]
        sin_hi = sin[tile_t, half:].to(torch.float32)[:, None, :]
        out[tile_t, tile_h, :half] = (lo * cos_lo - hi * sin_lo).to(out.dtype)
        out[tile_t, tile_h, half:] = (hi * cos_hi + lo * sin_hi).to(out.dtype)
    return out


def scaled_channel_rows(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Each ``[channels, n]`` matrix divided by its row sums, times a
    per-channel scale: every value carries the channel slice beside the
    reduced row, and the channel slice, which nothing reduces, is a loaded
    vector's stride-1 dim."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        values = x[tile, :, :].to(torch.float32)
        normed = values / values.sum(-1, keepdim=True)
        scaled = normed * scale[:].to(torch.float32)[None, :, None]
        out[tile, :, :] = scaled.to(out.dtype)
    return out


def column_norm(x: torch.Tensor) -> torch.Tensor:
    """Each column divided by its mean: the reduced full slice walks the
    strided dim."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(1)):
        col = x[:, tile].to(torch.float32)
        out[:, tile] = (col / col.mean(0, keepdim=True)).to(out.dtype)
    return out


def row_sum_scales_computed_tail(x: torch.Tensor) -> torch.Tensor:
    """The second half of each row times the row's sum, sliced from a bound
    the kernel computes: ``n`` is a value of its graph."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        n = row.size(1) // 2
        total = row.sum(-1, keepdim=True)
        out[tile, n:] = (x[tile, n:].to(torch.float32) * total).to(out.dtype)
    return out


def row_mean_and_max(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The row's mean and max stored to vectors, beside the scaled row."""
    rows = x.size(0)
    out = torch.empty_like(x)
    mean = torch.empty([rows], dtype=torch.float32, device=x.device)
    amax = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        row_mean = row.mean(-1)
        row_max = row.amax(-1)
        mean[tile] = row_mean
        amax[tile] = row_max
        out[tile, :] = ((row - row_mean[:, None]) / row_max[:, None]).to(out.dtype)
    return out, mean, amax


def rmsnorm_in_place(x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm stored over its input."""
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt((row * row).mean(-1, keepdim=True) + 1e-6)
        x[tile, :] = (row * rstd).to(x.dtype)
    return x


def rmsnorm_first_quarter(x: torch.Tensor) -> torch.Tensor:
    """The first quarter of each row, scaled by the whole row's RMS: a
    quarter-row slice swept beside the reduced row."""
    rows, dim = x.shape
    quarter = dim // 4
    out = torch.empty([rows, quarter], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile, :] = (x[tile, :quarter].to(torch.float32) * rstd).to(out.dtype)
    return out


def row_sums(x: torch.Tensor) -> torch.Tensor:
    """A row sum alone: no load or store of the row after the reduction."""
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return out


def rows_scaled_by_another_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The rows of ``y`` times the sums of the rows of ``x``, a tensor of
    another width: the swept slice is a row of another tensor."""
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        total = x[tile, :].to(torch.float32).sum(-1, keepdim=True)
        out[tile, :] = (y[tile, :].to(torch.float32) * total).to(out.dtype)
    return out


def row_argmax_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, dim = x.shape
    out = torch.empty([rows, dim], dtype=x.dtype, device=x.device)
    idx = torch.empty([rows], dtype=torch.int64, device=x.device)
    for tile in hl.tile(rows):
        row = x[tile, :].to(torch.float32)
        idx[tile] = torch.argmax(row, dim=-1)
        out[tile, :] = (row / row.amax(-1, keepdim=True)).to(out.dtype)
    return out, idx


def matmul_beside_row_norm(
    x: torch.Tensor, w: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A projection ``x @ w`` and a row normalization of ``z`` in one tile."""
    rows = z.size(0)
    out = torch.empty_like(z)
    proj = torch.empty([rows, w.size(1)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(rows):
        proj[tile, :] = x[tile, :].to(torch.float32) @ w[:, :]
        row = z[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    return out, proj


def grid_row_norm(x: torch.Tensor) -> torch.Tensor:
    """A row reduction over ``hl.grid``: no tile axis indexes the rows."""
    out = torch.empty_like(x)
    for i in hl.grid(x.size(0)):
        row = x[i, :].to(torch.float32)
        out[i, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
    return out


def row_sums_over_a_scale_matrix(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Each row's sum times the row of ``scale``, stored across a third dim
    as wide as the reduced row: the store spans the row block and the block
    of ``scale``'s rows, and its value carries only the second."""
    rows, dim = x.shape
    out = torch.empty([rows, scale.size(1), dim], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        total = x[tile, :].to(torch.float32).sum(-1, keepdim=True)
        scaled = total * scale[tile, :].to(torch.float32)
        out[tile, :, :] = scaled[:, :, None].to(out.dtype)
    return out


def row_sums_added_over_a_scale_matrix(
    x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``row_sums_over_a_scale_matrix`` with an atomic for its store: the
    atomic's subscript spans the row block and the block of ``scale``'s
    rows."""
    for tile in hl.tile(x.size(0)):
        total = x[tile, :].to(torch.float32).sum(-1, keepdim=True)
        scaled = total * scale[tile, :].to(torch.float32)
        hl.atomic_add(out, [tile, slice(None), slice(None)], scaled[:, :, None])
    return out


def row_sums_of_literal_copies(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The rows of ``y`` times the sums of 64 copies of the rows of ``x``:
    the copies carry the row beside a literal dim of ``y``'s width."""
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        copies = row[:, None, :].expand(-1, 64, -1)
        out[tile, :] = (copies.sum(-1) * y[tile, :].to(torch.float32)).to(out.dtype)
    return out


def row_sums_of_outer_products(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Each row of ``x`` times each element of the row of ``y``, summed over
    the row of ``x``: the products carry both rows as dims."""
    out = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        products = row[:, None, :] * y[tile, :].to(torch.float32)[:, :, None]
        out[tile, :] = products.sum(-1).to(out.dtype)
    return out


def row_sums_scale_outer_products(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """The outer product of the rows of ``a`` and ``b`` times the sum of the
    row of ``x``: the rows of ``a`` and ``b`` meet each other in the
    product, and neither meets the reduced row."""
    out = torch.empty([x.size(0), a.size(1), b.size(1)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        total = x[tile, :].to(torch.float32).sum(-1)
        outer = (
            a[tile, :].to(torch.float32)[:, :, None]
            * b[tile, :].to(torch.float32)[:, None, :]
        )
        out[tile, :, :] = (outer * total[:, None, None]).to(out.dtype)
    return out


def rmsnorm_heads_of_128_batches(x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm over each ``(batch, head)`` row, the batch axis fixed at
    128 batches a CTA."""
    out = torch.empty_like(x)
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[128, None]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile_b, tile_h, :] = (row * rstd).to(out.dtype)
    return out


def rmsnorm_heads_of_one_batch(x: torch.Tensor) -> torch.Tensor:
    """``rmsnorm_heads_of_128_batches`` with the batch axis fixed at one
    batch a CTA."""
    out = torch.empty_like(x)
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[1, None]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile_b, tile_h, :] = (row * rstd).to(out.dtype)
    return out


def rmsnorm_four_heads_a_cta(x: torch.Tensor) -> torch.Tensor:
    """``rmsnorm_heads_of_128_batches`` with the head axis, the innermost,
    fixed at four heads a CTA instead."""
    out = torch.empty_like(x)
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[None, 4]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile_b, tile_h, :] = (row * rstd).to(out.dtype)
    return out


def rmsnorm_heads_of_batches(x: torch.Tensor, batches: int) -> torch.Tensor:
    """``rmsnorm_heads_of_128_batches`` with the batches a CTA an argument."""
    out = torch.empty_like(x)
    for tile_b, tile_h in hl.tile([x.size(0), x.size(1)], block_size=[batches, None]):
        row = x[tile_b, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile_b, tile_h, :] = (row * rstd).to(out.dtype)
    return out


def row_norm_then_four_columns_a_tile(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A row normalization, then a device loop over the columns of ``y`` in
    tiles fixed at four columns."""
    out = torch.empty_like(x)
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        out[tile, :] = (row / row.sum(-1, keepdim=True)).to(out.dtype)
        for tile_c in hl.tile(y.size(1), block_size=4):
            doubled[tile, tile_c] = y[tile, tile_c] * 2
    return out, doubled


def clock_rows(x: torch.Tensor) -> torch.Tensor:
    """Each row's clock stamps minus their maximum: the stamps come from an
    impure inline assembly, which the store reads again after the
    reduction."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        stamp = hl.inline_asm_elementwise(
            "mov.u32 $0, %clock;", "=r", [x[tile, :]], torch.int32, False, 1
        ).to(torch.float32)
        out[tile, :] = (stamp - stamp.amax(-1, keepdim=True)).to(out.dtype)
    return out


def rmsnorm_rows_of_many(x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm whose row tile takes a registered block size of 128 rows
    or more."""
    out = torch.empty_like(x)
    block_size = hl.register_block_size(128, 256)
    for tile in hl.tile(x.size(0), block_size=block_size):
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt((row * row).mean(-1, keepdim=True) + 1e-6)
        out[tile, :] = (row * rstd).to(out.dtype)
    return out


def every_other_element_norm(x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm over every other element of each row: the reduced row is a
    stepped slice, at stride 2 of a contiguous tensor."""
    rows, dim = x.shape
    out = torch.empty([rows, dim // 2], dtype=x.dtype, device=x.device)
    for tile in hl.tile(rows):
        row = x[tile, ::2].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        out[tile, :] = (row * rstd).to(out.dtype)
    return out


def rmsnorm_rotate_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """An RMSNorm over each row, then a rotation of its interleaved pairs:
    the stepped slices ``x[tile, ::2]`` and ``x[tile, 1::2]`` walk the half
    block, swept beside the contiguous reduced row."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-6)
        even = x[tile, ::2].to(torch.float32) * rstd
        odd = x[tile, 1::2].to(torch.float32) * rstd
        cos_half = cos[tile, :].to(torch.float32)
        sin_half = sin[tile, :].to(torch.float32)
        out[tile, ::2] = (even * cos_half - odd * sin_half).to(out.dtype)
        out[tile, 1::2] = (odd * cos_half + even * sin_half).to(out.dtype)
    return out


def doubled_rows(x: torch.Tensor) -> torch.Tensor:
    """Pointwise over a full-slice row, which registers a reduction loop
    that nothing reduces."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = (x[tile, :].to(torch.float32) * 2).to(out.dtype)
    return out


def row_over_half_sum(x: torch.Tensor) -> torch.Tensor:
    """Each row divided by the sum of its first half: the row comes first, so
    it holds the kernel's reduction loop, but only the half is reduced."""
    half = x.size(1) // 2
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        row = x[tile, :].to(torch.float32)
        total = x[tile, :half].to(torch.float32).sum(-1, keepdim=True)
        out[tile, :] = (row / total).to(out.dtype)
    return out


def doubled_elements(x: torch.Tensor) -> torch.Tensor:
    """Pointwise over a vector: no full slice, so no reduction loop."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2
    return out


@functools.cache
def _bound(
    fn: Callable[..., object],
    *shapes: tuple[int, ...] | int,
    dtype: torch.dtype = torch.bfloat16,
    fp32: tuple[int, ...] = (),
    fp64: tuple[int, ...] = (),
    every_other: tuple[int, ...] = (),
    static_shapes: bool = True,
) -> BoundKernel:
    """``fn`` bound on CPU, once a module, to ``dtype`` tensors of
    ``shapes`` (an int is passed as it is); the arguments at the positions
    ``fp32`` are float32, those at ``fp64`` float64, and those at
    ``every_other`` are views of every other element of rows twice as
    long."""

    def argument(index: int, shape: tuple[int, ...] | int) -> torch.Tensor | int:
        if isinstance(shape, int):
            return shape
        kind = dtype
        if index in fp32:
            kind = torch.float32
        elif index in fp64:
            kind = torch.float64
        if index in every_other:
            return torch.empty(*shape[:-1], 2 * shape[-1], dtype=kind)[..., ::2]
        return torch.empty(*shape, dtype=kind)

    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="full",
        ignore_warnings=[helion.exc.FastMathUnused],
    )
    with cpu_target():
        return cpu_bind(kernel, tuple(starmap(argument, enumerate(shapes))))


def _layernorm(rows: int, hidden: int, **options: object) -> BoundKernel:
    return _bound(
        residual_layernorm,
        (rows, hidden),
        (rows, hidden),
        (hidden,),
        (hidden,),
        **options,
    )


def _rotate(tokens: int, heads: int, dim: int) -> BoundKernel:
    """``rmsnorm_rotate`` on bf16 rows with an fp32 weight."""
    return _bound(
        rmsnorm_rotate,
        (tokens, heads, dim),
        (dim,),
        (tokens, dim),
        (tokens, dim),
        fp32=(1,),
    )


@functools.cache
def _render(bound: BoundKernel, seed: Config) -> str:
    """The code of ``seed`` as the search compiles it: normalized, which must
    keep everything the seed spells."""
    generation = bound.config_spec.create_config_generation()
    _, normalized = generation.canonicalize_flat(generation.flatten(seed))
    for key, value in seed.config.items():
        assert normalized.config[key] == value, key
    return bound.to_code(normalized)


def _seeds(
    heuristic: type[
        CuteSubwarpRowHeuristic
        | CuteFlooredWarpRowHeuristic
        | CuteMaskedTailRowHeuristic
        | CuteSliceSweepPacketHeuristic
        | CuteLaneSplitRowHeuristic
    ],
    bound: BoundKernel,
) -> list[Config]:
    """The seeds of ``heuristic`` for ``bound``, each of them rendered: a
    seed the lowering refuses is a search slot spent on nothing, and the
    families emit two seeds at most, so no test checks a seed's values
    without compiling it."""
    seeds = heuristic.get_seed_configs(bound.env, bound.host_function.device_ir)
    for seed in seeds:
        _render(bound, seed)
    return seeds


def _values(seeds: list[Config], key: str) -> list[object]:
    return [seed.config[key] for seed in seeds]


def _scalar_accesses(code: str) -> int:
    """Element loads and stores of ``code`` that are not packets."""
    return len(re.findall(r"\)\.load\(\)|\)\.store\(", code))


def test_row_threads_follow_the_packet_count() -> None:
    row_threads = CuteSubwarpRowHeuristic.row_threads
    # 16 packets: one per thread.  24 and 48 packets: 8 and 16 threads of
    # three.  32-byte packets halve the count.
    assert row_threads(128, 8) == 16
    assert row_threads(192, 8) == 8
    assert row_threads(384, 8) == 16
    assert row_threads(96, 4) == 8
    assert row_threads(128, 8, 32) == 8
    assert row_threads(192, 8, 32) == 4
    # Whole warps of 16-byte packets belong to the warp rows at either
    # packet width.
    assert row_threads(256, 8) is None
    assert row_threads(768, 8) is None
    assert row_threads(256, 8, 32) is None
    # A thread walks at most 16 16-byte packets: 15 on each of 16 or 4
    # threads, not 17, 25 (8 threads) or 125 (4 threads).
    assert row_threads(1920, 8) == 16
    assert row_threads(480, 8) == 4
    assert row_threads(960, 4) == 16
    assert row_threads(2176, 8) is None
    assert row_threads(1088, 4) is None
    assert row_threads(1600, 8) is None
    assert row_threads(4000, 8) is None
    # 32-byte packets count twice: seven on 8 threads, not nine or 15.
    assert row_threads(896, 8, 32) == 8
    assert row_threads(1152, 8, 32) is None
    assert row_threads(1920, 8, 32) is None
    # No whole number of packets, or too few threads to share a row.
    assert row_threads(100, 8) is None
    assert row_threads(72, 8, 32) is None
    assert row_threads(200, 8) is None
    assert row_threads(48, 8) is None
    assert row_threads(96, 8, 32) is None


def test_non_power_of_two_row_below_the_occupancy_floor_gets_seeds() -> None:
    # 25216 rows exceed the grid budget at one row per CTA, so the search
    # floor is two rows and the one-row heuristics decline the kernel.
    bound = _layernorm(25216, 192)
    spec = bound.config_spec
    assert spec.block_sizes[0].autotuner_min == 2
    assert not CuteReductionTileHeuristic.is_eligible(
        bound.env, bound.host_function.device_ir
    )
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    # 8 threads of three 16-byte packets, then 4 threads of three 32-byte
    # packets, 8 rows per CTA.
    assert _values(seeds, "block_sizes") == [[8], [8]]
    assert _values(seeds, "num_threads") == [[8, 8], [8, 4]]
    for seed, width in zip(seeds, (8, 16), strict=True):
        config = seed.config
        assert config["reduction_loops"] == [192]
        assert (
            spec.cute_vector_widths.config_get(config["cute_vector_widths"], 1) == width
        )
        assert spec.cute_lane_layouts.config_get(config["cute_lane_layouts"], 1) == (
            "strided"
        )
        assert config["cute_reduction_reloads"] == ["register_prefetch"]
    assert spec.compiler_seed_configs == seeds
    assert spec.autotuner_heuristics == [CuteSubwarpRowHeuristic.name]
    # The search takes the seeds as they are.
    pairs = spec.create_config_generation().seed_flat_config_pairs()
    assert [normalized.config["num_threads"] for _, normalized in pairs] == [
        [8, 8],
        [8, 4],
    ]
    code = _render(bound, seeds[1])
    assert "block=(4, 8, 1)" in code
    assert "VectorType.get([16]" in code
    assert _scalar_accesses(code) == 0
    # Strided lanes: a thread's three packets are a third of the row apart.
    assert "thread_idx()[0]) * 16 + cutlass.Int32(reduction_lane_1) * 64" in code


def test_32_byte_packets_need_sm100(monkeypatch: pytest.MonkeyPatch) -> None:
    bound = _layernorm(25216, 192)
    spec = bound.config_spec
    assert spec.target_device_capability == (10, 0)
    monkeypatch.setattr(spec, "target_device_capability", (9, 0))
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "num_threads") == [[8, 8]]


def test_short_power_of_two_row_gets_one_packet_per_thread() -> None:
    bound = _layernorm(2048, 128)
    device_ir = bound.host_function.device_ir
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(bound.env, device_ir) == []
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "num_threads") == [[8, 16], [8, 8]]
    assert _values(seeds, "reduction_loops") == [[128], [128]]
    assert all(seed in bound.config_spec.compiler_seed_configs for seed in seeds)


def test_rows_per_cta_start_at_the_occupancy_floor() -> None:
    # The seeds ask for 8 rows per CTA; this grid's floor is 16.
    bound = _layernorm(151552, 128)
    spec = bound.config_spec
    assert spec.block_sizes[0]._fragment(spec).low == 16
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "block_sizes") == [[16], [16]]
    assert _values(seeds, "num_threads") == [[16, 16], [16, 8]]


def test_rows_per_cta_stop_at_the_axis() -> None:
    # Four heads: the innermost tile axis holds four rows at most.
    bound = _rotate(64, 4, 128)
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "block_sizes") == [[1, 4], [1, 4]]
    assert _values(seeds, "num_threads") == [[0, 4, 16, 16], [0, 4, 8, 8]]


def test_rows_per_cta_go_to_the_innermost_tile_axis() -> None:
    # The inner tile axis, block 0, takes a block size registered before the
    # loop, so it is the first block size and the batch axis the last.
    bound = _bound(softmax_rows_inner_block_size_first, (64, 8, 128))
    spec = bound.config_spec
    assert [item.block_id for item in spec.block_sizes] == [0, 1]
    assert bound.host_function.device_ir.grid_block_ids == [[1, 0]]
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "block_sizes") == [[8, 1], [8, 1]]
    assert _values(seeds, "num_threads") == [[8, 0, 16], [8, 0, 8]]
    assert "block=(16, 8, 1)" in _render(bound, seeds[0])


def test_cta_holds_a_warp_of_threads_and_no_more_than_it_launches() -> None:
    # One head: the innermost tile axis holds one row, of 16 or 8 threads,
    # which is less than a warp at either packet width.
    one_head = _rotate(4096, 1, 128)
    spec = one_head.config_spec
    assert spec.block_sizes[1]._fragment(spec).high == 1
    assert CuteSubwarpRowHeuristic.row_threads(128, 8) == 16
    assert _seeds(CuteSubwarpRowHeuristic, one_head) == []
    assert CuteSubwarpRowHeuristic.name not in spec.autotuner_heuristics
    # A row tile of 128 rows or more: 128 rows of 16 threads are twice the
    # 1024 threads a CTA launches, 128 rows of 8 threads fill it.
    many = _bound(rmsnorm_rows_of_many, (4096, 128))
    spec = many.config_spec
    assert spec.block_sizes[0]._fragment(spec).low == 128
    seeds = _seeds(CuteSubwarpRowHeuristic, many)
    assert _values(seeds, "block_sizes") == [[128]]
    assert _values(seeds, "num_threads") == [[128, 8]]
    assert "block=(8, 128, 1)" in _render(many, seeds[0])


def test_one_row_plan_serves_the_four_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The warp rows above the floor and their twins with packets on the
    # quarter rows: two of the four families, in the order they are
    # appended.
    shapes = [(25216, 256)]
    bound = _bound(rmsnorm_first_quarter, *shapes)
    env, device_ir = bound.env, bound.host_function.device_ir
    families = row_packet_seed_configs(env, device_ir)
    assert families == [
        (heuristic.name, heuristic.get_seed_configs(env, device_ir))
        for heuristic in ROW_HEURISTICS
    ]
    assert [len(seeds) for _, seeds in families] == [2, 0, 0, 2]
    assert bound.config_spec.autotuner_heuristics[-2:] == [
        CuteFlooredWarpRowHeuristic.name,
        CuteSliceSweepPacketHeuristic.name,
    ]
    # A bind makes the plan once.
    plans = []
    row_plan = cute_row_packets._row_plan
    monkeypatch.setattr(
        cute_row_packets,
        "_row_plan",
        lambda env, device_ir: plans.append(row_plan(env, device_ir)) or plans[-1],
    )
    rebound = _bound.__wrapped__(rmsnorm_first_quarter, *shapes)
    assert len(plans) == 1
    assert rebound.config_spec.compiler_seed_configs == (
        bound.config_spec.compiler_seed_configs
    )


def test_rows_of_whole_warps_of_packets_keep_the_warp_rows() -> None:
    bound = _layernorm(2048, 256)
    assert _seeds(CuteSubwarpRowHeuristic, bound) == []
    assert CuteSubwarpRowHeuristic.name not in bound.config_spec.autotuner_heuristics
    # No slice is swept beside this row: the warp rows get no twin.  One row
    # per CTA is searchable, so ``CuteReductionTileHeuristic`` seeded them.
    assert _seeds(CuteSliceSweepPacketHeuristic, bound) == []
    assert _seeds(CuteFlooredWarpRowHeuristic, bound) == []
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(
        bound.env, bound.host_function.device_ir
    )
    # The seeds open no knob: the search space is the shorter row's.
    short = _layernorm(2048, 128)
    assert (
        bound.config_spec.structural_fingerprint()
        == short.config_spec.structural_fingerprint()
    )


def test_warp_rows_above_the_occupancy_floor_are_seeded() -> None:
    # As at 192 wide, 25216 rows put the search floor at two rows, so
    # ``CuteReductionTileHeuristic`` declines; its warp rows hold 2 and 4.
    bound = _layernorm(25216, 256)
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert spec.block_sizes[0]._fragment(spec).low == 2
    assert not CuteReductionTileHeuristic.is_eligible(bound.env, device_ir)
    seeds = _seeds(CuteFlooredWarpRowHeuristic, bound)
    assert seeds == CuteReductionTileHeuristic.warp_rows_seed_configs(
        bound.env, device_ir
    )
    assert _values(seeds, "block_sizes") == [[2], [4]]
    assert _values(seeds, "num_threads") == [[2, 32], [4, 32]]
    assert spec.compiler_seed_configs == seeds
    assert spec.autotuner_heuristics == [CuteFlooredWarpRowHeuristic.name]


def test_floored_warp_rows_read_the_row_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One row per CTA is searchable here.  The floored family reads that
    # floor itself: another reason for ``CuteReductionTileHeuristic`` to
    # decline a row kernel, a gate it has or gains, is no floor.
    bound = _layernorm(2048, 256)
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert spec.block_sizes[0]._fragment(spec).low == 1
    monkeypatch.setattr(
        CuteReductionTileHeuristic,
        "is_eligible",
        classmethod(lambda cls, env, device_ir: False),
    )
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(bound.env, device_ir)
    assert _seeds(CuteFlooredWarpRowHeuristic, bound) == []


def test_warp_rows_get_a_twin_with_packets_on_the_swept_slices() -> None:
    # Block ids: 0 tokens, 1 heads, 2 the 256-wide row, 3 its 128-wide halves.
    bound = _rotate(64, 4, 256)
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert slice_packet_entries(bound.env, device_ir, 2, 32) == ({3: 32}, {3: 4})
    warp_rows = CuteReductionTileHeuristic.warp_rows_seed_configs(bound.env, device_ir)
    twins = _seeds(CuteSliceSweepPacketHeuristic, bound)
    assert _values(twins, "block_sizes") == [[1, 2], [1, 4]]
    for seed, twin in zip(warp_rows, twins, strict=True):
        assert seed.config["num_threads"] == twin.config["num_threads"]
        widths = twin.config["cute_vector_widths"]
        assert spec.cute_vector_widths.config_get(widths, 2) == 8
        assert spec.cute_vector_widths.config_get(widths, 3) == 4
    # Appended after every older seed, whose list is unchanged.
    seeds = spec.compiler_seed_configs
    assert seeds[-2:] == twins
    assert all(seed in seeds[:-2] for seed in warp_rows)
    assert spec.autotuner_heuristics[-1] == CuteSliceSweepPacketHeuristic.name
    # The warp-rows seed sweeps the halves as scalars; its twin as packets on
    # the same 32 threads a row.
    scalar = _render(bound, warp_rows[0])
    packets = _render(bound, twins[0])
    assert "block=(32, 2, 1)" in scalar
    assert "block=(32, 2, 1)" in packets
    assert _scalar_accesses(scalar) > 0
    assert _scalar_accesses(packets) == 0
    assert "VectorType.get([4]" in packets


def test_slice_share_is_one_vector_within_the_packet() -> None:
    # Block 3: the 96-wide halves of a 192-wide bf16 row, beside which the
    # fp32 weight is sliced, so a 16-byte packet is four elements.
    bound = _rotate(64, 4, 192)
    device_ir = bound.host_function.device_ir

    def entries(threads: int, packet_bytes: int = 16) -> object:
        return slice_packet_entries(bound.env, device_ir, 2, threads, packet_bytes)

    assert entries(24) == ({3: 24}, {3: 4})
    # Three elements a thread are no vector; 40 threads do not divide the
    # slice; one element a thread is no packet.
    assert entries(32) is None
    assert entries(40) is None
    assert entries(96) is None
    # Eight elements a thread are 32 bytes of the weight.
    assert entries(12) is None
    assert entries(12, 32) == ({3: 12}, {3: 8})
    # The row's 8 and 4 threads leave the halves 12 and 24 elements a
    # thread: no packet, so no seed.
    assert CuteSubwarpRowHeuristic.row_threads(192, 8) == 8
    assert _seeds(CuteSubwarpRowHeuristic, bound) == []


def test_warp_rows_without_slice_packets_get_no_twin() -> None:
    # 256-wide halves on 32 row threads are 8 elements a thread: one 16-byte
    # packet of the bf16 row, but 32 bytes of the fp32 weight sliced beside
    # it.
    bound = _rotate(64, 4, 512)
    device_ir = bound.host_function.device_ir
    assert CuteReductionTileHeuristic.warp_rows_seed_configs(bound.env, device_ir)
    assert slice_packet_entries(bound.env, device_ir, 2, 32) is None
    assert _seeds(CuteSliceSweepPacketHeuristic, bound) == []
    assert (
        CuteSliceSweepPacketHeuristic.name not in bound.config_spec.autotuner_heuristics
    )


def test_subwarp_rows_carry_the_slice_packets() -> None:
    bound = _rotate(64, 8, 128)
    spec = bound.config_spec
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "block_sizes") == [[1, 8], [1, 8]]
    assert _values(seeds, "num_threads") == [[0, 8, 16, 16], [0, 8, 8, 8]]
    for seed, (row, half) in zip(seeds, ((8, 4), (16, 8)), strict=True):
        widths = seed.config["cute_vector_widths"]
        assert spec.cute_vector_widths.config_get(widths, 2) == row
        assert spec.cute_vector_widths.config_get(widths, 3) == half
    assert _seeds(CuteSliceSweepPacketHeuristic, bound) == []
    code = _render(bound, seeds[1])
    assert "block=(8, 8, 1)" in code
    assert _scalar_accesses(code) == 0
    # Left to the backend, the halves take more threads than the row's 8 and
    # an axis of their own, where they run scalar.
    threads = list(seeds[1].config["num_threads"])
    threads[spec.num_threads.block_id_to_index(3)] = 0
    own_axis = bound.to_code(
        Config.from_dict(seeds[1].config | {"num_threads": threads})
    )
    assert _scalar_accesses(own_axis) > 0
    # The seeds open no knob: the search space is the wider row's.
    wide = _rotate(64, 4, 256)
    assert spec.structural_fingerprint() == wide.config_spec.structural_fingerprint()


def test_full_slices_meet_as_the_lowering_reads_them() -> None:
    # ``x[tile, :, :]`` holds the channel slice beside the reduced row, so
    # the two need the cross product of their coordinates and thread axes
    # of their own.
    bound = _bound(scaled_channel_rows, (64, 64, 256), (64,))
    spec = bound.config_spec
    fact = spec.reduction_kernel_fact
    assert fact is not None
    (row,) = {descriptor.block_id for descriptor in fact.reductions}
    (channels,) = spec.reduction_block_ids - {row}
    assert spec.cute_static_reduction_extents == {channels: 64, row: 256}
    assert _full_slices_meet(bound.env, bound.host_function.device_ir)
    for heuristic in ROW_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    # The half rows of the rotation, block 3, meet the row, block 2, in no
    # value and no access: they are swept beside it, on its thread axis.
    rotate = _rotate(64, 4, 256)
    assert rotate.config_spec.cute_static_reduction_extents == {2: 256, 3: 128}
    assert not _full_slices_meet(rotate.env, rotate.host_function.device_ir)


def test_tile_axis_fixed_at_one_block_keeps_the_seeds() -> None:
    # One batch a CTA adds no threads: the CTA is the seeds' 8 rows of 8 and
    # of 4 threads.
    bound = _bound(rmsnorm_heads_of_one_batch, (256, 16, 192))
    spec = bound.config_spec
    assert spec.cute_tile_fixed_block_sizes == {0: 1}
    assert [item.block_id for item in spec.block_sizes] == [1]
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "block_sizes") == [[8], [8]]
    assert _values(seeds, "num_threads") == [[8, 8], [8, 4]]
    assert "block=(8, 8, 1)" in _render(bound, seeds[0])
    assert "block=(4, 8, 1)" in _render(bound, seeds[1])


def test_a_thread_walks_at_most_sixteen_packets() -> None:
    # 1920 elements are 15 packets on each of 16 threads.  As 32-byte
    # packets they would be 15 on each of 8 threads, 480 bytes a thread: no
    # seed at that width.
    bound = _layernorm(2048, 1920)
    spec = bound.config_spec
    seeds = _seeds(CuteSubwarpRowHeuristic, bound)
    assert _values(seeds, "num_threads") == [[8, 16]]
    widths = seeds[0].config["cute_vector_widths"]
    assert spec.cute_vector_widths.config_get(widths, 1) == 8
    code = _render(bound, seeds[0])
    assert "block=(16, 8, 1)" in code
    assert code.count("for reduction_lane_1 in range(15)") == 3
    # The row is loaded once and its packets stay in registers for the two
    # later sweeps.
    assert code.count("cute.arch.load(x.iterator") == 1
    assert "_fuse_cache_0[" in code
    # 1600 elements would be 25 packets on each of 8 threads, 4000 elements
    # 125 on each of 4: no seed, at either packet width.
    for hidden in (1600, 4000):
        wide = _layernorm(2048, hidden)
        assert _seeds(CuteSubwarpRowHeuristic, wide) == []
        assert CuteSubwarpRowHeuristic.name not in wide.config_spec.autotuner_heuristics


def test_rows_no_packet_tiles_take_trips_of_packets() -> None:
    # 196 bf16 elements are 24.5 16-byte packets and 49 8-byte ones: no
    # power-of-two thread count tiles them, so the subwarp rows decline.
    # The row takes 8-byte packets on 16 threads, a trip of 64 elements,
    # three trips plus one packet; 8 rows per CTA.
    bound = _layernorm(2048, 196)
    spec = bound.config_spec
    assert _seeds(CuteSubwarpRowHeuristic, bound) == []
    (seed,) = _seeds(CuteMaskedTailRowHeuristic, bound)
    assert CuteMaskedTailRowHeuristic.name in spec.autotuner_heuristics
    values = seed.config
    assert values["block_sizes"] == [8]
    assert values["num_threads"] == [8, 16]
    assert values["reduction_loops"] == [64]
    assert spec.cute_vector_widths.config_get(values["cute_vector_widths"], 1) == 4
    code = _render(bound, seed)
    assert "block=(16, 8, 1)" in code
    # Over two tile axes the rows' axis also leads the CTA order: a second
    # seed, the first with the reversed loop order.
    over_tiles = _bound(softmax_rows_over_two_tiles, (32, 64, 196))
    first, reversed_order = _seeds(CuteMaskedTailRowHeuristic, over_tiles)
    assert first.config["block_sizes"] == [1, 8]
    assert "loop_orders" not in first.config
    assert reversed_order.config == {**first.config, "loop_orders": [[1, 0]]}


def _softmax_backward(rows: int, keys: int) -> BoundKernel:
    """``masked_softmax_backward_rows`` over [4, 2, rows] rows of ``keys``,
    bound on CPU."""
    bf16, fp32, flag = torch.bfloat16, torch.float32, torch.bool
    args = tuple(
        torch.empty(shape, dtype=dtype)
        for shape, dtype in (
            ((4, 2, rows, keys), bf16),
            ((4, 2, rows, keys), flag),
            ((4, 2, rows, keys), bf16),
            ((4, 2, rows), fp32),
            ((4, 2, rows), fp32),
            ((4, rows, keys), flag),
            ((4, 2, rows, keys), bf16),
        )
    )
    kernel = helion.kernel(
        masked_softmax_backward_rows,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        ignore_warnings=[helion.exc.FastMathUnused],
    )
    with cpu_target():
        return cpu_bind(kernel, args)


def test_lane_split_rows_take_rows_of_threads() -> None:
    # A softmax backward's rows of 128 keys: no reduction loop, so no other
    # row family seeds them.  8 rows of the innermost tile axis per CTA, with
    # 16 threads of 2-element packets on the row (4 trips of 32) and with 8
    # threads of 8-element packets (2 trips of 64).
    bound = _softmax_backward(128, 128)
    spec = bound.config_spec
    assert not spec.reduction_loops
    for heuristic in ROW_HEURISTICS:
        assert _seeds(heuristic, bound) == []
    seeds = _seeds(CuteLaneSplitRowHeuristic, bound)
    assert CuteLaneSplitRowHeuristic.name in spec.autotuner_heuristics
    (row,) = spec.reduction_block_ids
    assert _values(seeds, "block_sizes") == [[1, 1, 8], [1, 1, 8]]
    assert [
        spec.num_threads.config_get(seed.config["num_threads"], row) for seed in seeds
    ] == [16, 8]
    assert [
        spec.cute_vector_widths.config_get(seed.config["cute_vector_widths"], row)
        for seed in seeds
    ] == [2, 8]
    # Rows of 512 keys take both layouts (16 packets of 2 per thread at
    # most); rows of 4096 would give each thread 128 or 64 packets, past the
    # subwarp rows' MAX_THREAD_PACKETS, and take none.
    assert len(_seeds(CuteLaneSplitRowHeuristic, _softmax_backward(16, 512))) == 2
    assert _seeds(CuteLaneSplitRowHeuristic, _softmax_backward(16, 4096)) == []
    # A row that a reduction loop rolls is the row plan's: rows of 96 keys
    # (whose kernel has the loop) and the LayerNorm's.
    rolled = _softmax_backward(128, 96)
    assert rolled.config_spec.reduction_loops
    assert _seeds(CuteLaneSplitRowHeuristic, rolled) == []
    assert _seeds(CuteLaneSplitRowHeuristic, _layernorm(2048, 196)) == []


def test_rows_a_packet_tiles_take_no_trips() -> None:
    # 192 elements are 24 packets on 8 threads: the subwarp rows'.  49 are
    # no whole packets at any width, 4000 are 16 trips of 128 and more.
    # 1600 are 8 threads of 25 packets, past the subwarp rows' cap: no
    # subwarp seed, and no masked-tail one either.
    for hidden, subwarp in ((192, True), (49, False), (4000, False), (1600, False)):
        bound = _layernorm(2048, hidden)
        assert bool(_seeds(CuteSubwarpRowHeuristic, bound)) == subwarp, hidden
        assert _seeds(CuteMaskedTailRowHeuristic, bound) == [], hidden


# Row kernels beyond the residual LayerNorm and the rotated RMSNorm, one case
# per structure and family: every seed a family emits for them is rendered
# (``_seeds``).  The values are the fired families' seed counts.
@pytest.mark.parametrize(
    ("kernel", "shapes", "options", "expected"),
    [
        pytest.param(
            softmax_rows, [(2048, 192)], {}, {"subwarp": 2}, id="two reductions"
        ),
        pytest.param(
            softmax_rows,
            [(25216, 256)],
            {},
            {"floored": 2},
            id="two reductions, floored grid",
        ),
        pytest.param(
            row_mean_and_max,
            [(2048, 384)],
            {},
            {"subwarp": 2},
            id="reductions stored to vectors",
        ),
        pytest.param(
            rmsnorm_in_place, [(2048, 128)], {}, {"subwarp": 2}, id="in-place rows"
        ),
        pytest.param(
            rmsnorm_first_quarter,
            [(25216, 256)],
            {},
            {"floored": 2, "twins": 2},
            id="quarter-row slice, floored grid",
        ),
        pytest.param(
            rmsnorm_first_quarter,
            [(2048, 128)],
            {"dtype": torch.float16},
            {"subwarp": 2},
            id="quarter-row slice, fp16",
        ),
        pytest.param(
            row_sums,
            [(2048, 96)],
            {"dtype": torch.float32},
            {"subwarp": 2},
            id="row sum alone, fp32",
        ),
        pytest.param(
            rows_scaled_by_another_sum,
            [(2048, 192), (2048, 64)],
            {},
            {"subwarp": 2},
            id="rows of another tensor swept",
        ),
    ],
)
def test_seeds_render_on_other_row_kernels(
    kernel: Callable[..., object],
    shapes: list[tuple[int, ...]],
    options: dict[str, object],
    expected: dict[str, int],
) -> None:
    bound = _bound(kernel, *shapes, **options)
    fact = bound.config_spec.reduction_kernel_fact
    assert fact is not None
    assert len({descriptor.block_id for descriptor in fact.reductions}) == 1
    fired = {
        label: len(_seeds(heuristic, bound))
        for label, heuristic in (
            ("floored", CuteFlooredWarpRowHeuristic),
            ("subwarp", CuteSubwarpRowHeuristic),
            ("masked tail", CuteMaskedTailRowHeuristic),
            ("twins", CuteSliceSweepPacketHeuristic),
        )
    }
    assert {label: count for label, count in fired.items() if count} == expected


def test_row_accesses_are_read_without_the_other_gates() -> None:
    # Called on its own, in any order with the plan's other gates.  A slice
    # from a bound the kernel computes is read as the graph spells it, a
    # node (taking its extent raises a TypeError), so it declines every
    # block.
    computed = _bound(row_sum_scales_computed_tail, (2048, 128))
    for block_id in computed.config_spec.reduction_block_ids:
        assert _row_is_no_run_of_packets(
            computed.env, computed.host_function.device_ir, block_id
        )
    # A row walked along a strided dim; a contiguous row.
    column = _bound(column_norm, (192, 192))
    assert _row_is_no_run_of_packets(column.env, column.host_function.device_ir, 1)
    rows = _layernorm(2048, 192)
    assert not _row_is_no_run_of_packets(rows.env, rows.host_function.device_ir, 1)
    # A row that is a stepped slice of a contiguous tensor.
    stepped = _bound(every_other_element_norm, (2048, 384))
    assert _row_is_no_run_of_packets(stepped.env, stepped.host_function.device_ir, 1)


def test_stepped_slices_of_the_half_block_leave_the_row_its_packets() -> None:
    # Block 1 is the contiguous 256-wide row and block 2 its interleaved
    # halves, ``x[tile, ::2]`` and ``x[tile, 1::2]``, which no load or store
    # of the row walks: the warp rows keep their twins.
    bound = _bound(rmsnorm_rotate_interleaved, (2048, 256), (2048, 128), (2048, 128))
    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert spec.cute_static_reduction_extents == {1: 256, 2: 128}
    assert not _row_is_no_run_of_packets(bound.env, device_ir, 1)
    assert _row_is_no_run_of_packets(bound.env, device_ir, 2)
    twins = _seeds(CuteSliceSweepPacketHeuristic, bound)
    assert _values(twins, "num_threads") == [[2, 32, 0], [4, 32, 0]]
    assert _values(twins, "cute_vector_widths") == [[8, 4, 1], [8, 4, 1]]
    assert all(seed in spec.compiler_seed_configs for seed in twins)
    # The row and the contiguous half rows of cos and sin load as packets;
    # the interleaved elements of x and out stay scalar.
    code = _render(bound, twins[0])
    assert "VectorType.get([8]" in code
    assert code.count("VectorType.get([4]") == 2
    assert _scalar_accesses(code) == 4


def test_slice_of_a_dtype_without_packets_gets_no_seed_at_either_width() -> None:
    # The 16-wide rows of ``y`` are swept beside the 128-wide bf16 row.  As
    # bf16 they are one 4-byte vector on each of the 8 threads of the
    # 32-byte seed (16 threads would leave them one element each).
    packets = _bound(rows_scaled_by_another_sum, (2048, 128), (2048, 16))
    device_ir = packets.host_function.device_ir
    assert slice_packet_entries(packets.env, device_ir, 1, 16) is None
    assert slice_packet_entries(packets.env, device_ir, 1, 8, 32) == ({2: 8}, {2: 2})
    seeds = _seeds(CuteSubwarpRowHeuristic, packets)
    assert _values(seeds, "num_threads") == [[8, 8, 8]]
    assert _values(seeds, "cute_vector_widths") == [[16, 2, 1]]
    # As fp64 they load no packets: no entry at 16 or at 32 bytes, where
    # the width of "no vector" times two would pass for one, and no seed.
    scalar = _bound(rows_scaled_by_another_sum, (2048, 128), (2048, 16), fp64=(1,))
    device_ir = scalar.host_function.device_ir
    assert scalar.config_spec.cute_static_reduction_extents == {1: 128, 2: 16}
    for threads, packet_bytes in ((16, 16), (8, 32), (4, 32)):
        assert (
            slice_packet_entries(scalar.env, device_ir, 1, threads, packet_bytes)
            is None
        )
    assert _seeds(CuteSubwarpRowHeuristic, scalar) == []
    assert CuteSubwarpRowHeuristic.name not in scalar.config_spec.autotuner_heuristics


# One kernel per gate of ``_row_plan``, each a row that would get seeds if
# its gate were dropped.  A case binds its kernel and checks the facts that
# make it the case it names.


def _reduced_blocks(bound: BoundKernel) -> set[int]:
    fact = bound.config_spec.reduction_kernel_fact
    assert fact is not None
    return {descriptor.block_id for descriptor in fact.reductions}


def _loop_blocks(bound: BoundKernel) -> list[int]:
    return [loop.block_id for loop in bound.config_spec.reduction_loops]


def _row_floor(bound: BoundKernel) -> int:
    spec = bound.config_spec
    return spec.block_sizes[len(spec.block_sizes) - 1]._fragment(spec).low


def _has_warp_rows(bound: BoundKernel) -> bool:
    """Whether ``CuteReductionTileHeuristic`` has warp rows for the kernel,
    which two of the families would read past their own gate."""
    return bool(
        CuteReductionTileHeuristic.warp_rows_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    )


def _second_root_loop() -> BoundKernel:
    # An hl.grid loop of its own before the row loop.  It has no block size,
    # so ``CuteReductionTileHeuristic`` seeds the rows; the families need
    # one root grid.
    bound = _bound(grid_first_elements_then_row_norm, (2048, 192), (2048,), fp32=(1,))
    assert bound.host_function.device_ir.grid_block_ids == [[0], [1]]
    assert CuteReductionTileHeuristic.is_eligible(
        bound.env, bound.host_function.device_ir
    )
    return bound


def _second_root_loop_with_a_block_size() -> BoundKernel:
    bound = _bound(
        flag_then_rmsnorm_rotate,
        (64, 4, 256),
        (256,),
        (64, 256),
        (64, 256),
        (64,),
        fp32=(1,),
    )
    assert len(bound.host_function.device_ir.grid_block_ids) == 2
    return bound


def _tile_loop_nested_in_the_row_loop() -> BoundKernel:
    # Block 2 tiles the columns of ``y`` in a device loop: the seeds would
    # put their rows per CTA on it.
    bound = _bound(row_norm_then_tiled_columns, (2048, 192), (2048, 64))
    assert bound.host_function.device_ir.grid_block_ids == [[0]]
    assert [item.block_id for item in bound.config_spec.block_sizes] == [0, 2]
    return bound


def _registered_block_size_that_no_loop_tiles() -> BoundKernel:
    # The rows are an ``hl.grid`` axis, block 1; the kernel's one block size
    # is the knob it divides by.
    bound = _bound(registered_scale_sum, (2048, 192))
    assert bound.host_function.device_ir.grid_block_ids == [[1]]
    assert [item.block_id for item in bound.config_spec.block_sizes] == [0]
    return bound


def _matmul_beside_the_row() -> BoundKernel:
    bound = _bound(matmul_beside_row_norm, (2048, 64), (64, 32), (2048, 192), fp32=(1,))
    assert bound.config_spec.matmul_facts
    assert len(bound.config_spec.reduction_loops) == 1
    return bound


def _no_tile_axis() -> BoundKernel:
    bound = _bound(grid_row_norm, (2048, 128))
    assert not bound.config_spec.block_sizes
    return bound


def _loop_that_nothing_reduces() -> BoundKernel:
    bound = _bound(doubled_rows, (2048, 128))
    assert len(bound.config_spec.reduction_loops) == 1
    return bound


def _loop_on_a_row_beside_the_reduced_one() -> BoundKernel:
    bound = _bound(row_over_half_sum, (2048, 128))
    assert _loop_blocks(bound) == [1]
    assert _reduced_blocks(bound) == {2}
    assert bound.config_spec.cute_static_reduction_extents == {1: 128, 2: 64}
    return bound


def _two_reduced_blocks(bound: BoundKernel) -> BoundKernel:
    # The loop's row, block 1, is reduced, and so is block 2: the row's
    # first half, or the 64-wide rows of ``y``.  The lowering refuses the
    # seeds' layout beside a second reduced block, at 8 rows of sub-warp
    # threads and at 2 and 4 warp rows alike.
    assert _reduced_blocks(bound) == {1, 2}
    assert _loop_blocks(bound) == [1]
    return bound


def _half_row_reduced_beside_the_row() -> BoundKernel:
    return _two_reduced_blocks(_bound(row_and_half_norms, (2048, 128)))


def _half_row_reduced_above_the_occupancy_floor() -> BoundKernel:
    bound = _two_reduced_blocks(_bound(row_and_half_norms, (25216, 256)))
    assert _row_floor(bound) == 2
    return bound


def _rows_of_a_second_tensor_reduced() -> BoundKernel:
    return _two_reduced_blocks(_bound(rows_over_two_sums, (2048, 192), (2048, 64)))


def _no_reduction_loop() -> BoundKernel:
    bound = _bound(doubled_elements, (2048,))
    assert not bound.config_spec.reduction_loops
    return bound


def _argmax_row() -> BoundKernel:
    bound = _bound(row_argmax_scale, (2048, 192))
    assert bound.config_spec.cute_indexed_reduction_block_ids
    return bound


def _loop_pinned_persistent() -> BoundKernel:
    # A sweep of the row's loop would read the clock a second time, so the
    # row stays persistent and its loop slot is pinned: normalization takes
    # the rolled trip out of a seed.
    bound = _bound(clock_rows, (2048, 128))
    (loop,) = bound.config_spec.reduction_loops
    assert loop.pinned
    return bound


def _loop_pinned_persistent_above_the_occupancy_floor() -> BoundKernel:
    bound = _bound(clock_rows, (25216, 256))
    (loop,) = bound.config_spec.reduction_loops
    assert loop.pinned
    assert _row_floor(bound) == 2
    assert _has_warp_rows(bound)
    return bound


def _argmax_row_above_the_occupancy_floor() -> BoundKernel:
    bound = _bound(row_argmax_scale, (25216, 256))
    assert _row_floor(bound) == 2
    # The floored warp rows would read these past the plan.
    assert _has_warp_rows(bound)
    return bound


def _column_reduction_on_a_square_tensor() -> BoundKernel:
    return _bound(column_norm, (192, 192))


def _rows_of_a_strided_view() -> BoundKernel:
    # ``x`` is every other element of 384-wide rows: no dim of it has stride
    # 1, so its rows are no runs of packets.
    return _layernorm(2048, 192, every_other=(0,))


def _row_sliced_with_a_step() -> BoundKernel:
    # ``x[tile, ::2]``: the 192-wide row is every other element of a
    # contiguous 384-wide one, whose stride-1 dim it walks at stride 2.
    bound = _bound(every_other_element_norm, (2048, 384))
    assert bound.config_spec.cute_static_reduction_extents == {1: 192}
    return bound


def _row_sliced_with_a_step_above_the_occupancy_floor() -> BoundKernel:
    bound = _bound(every_other_element_norm, (25216, 512))
    assert bound.config_spec.cute_static_reduction_extents == {1: 256}
    assert _row_floor(bound) == 2
    assert _has_warp_rows(bound)
    return bound


def _second_slice_meets_the_row(bound: BoundKernel) -> BoundKernel:
    # Block 1 is the 128-wide reduced row, which every load and store walks
    # at stride 1, and block 2 a 64-wide slice that nothing reduces.  The
    # lowering finds the two in one value or one access.
    assert bound.config_spec.cute_static_reduction_extents == {1: 128, 2: 64}
    assert _reduced_blocks(bound) == {1}
    assert _loop_blocks(bound) == [1]
    device_ir = bound.host_function.device_ir
    assert not _row_is_no_run_of_packets(bound.env, device_ir, 1)
    assert _full_slices_meet(bound.env, device_ir)
    return bound


def _store_across_the_row_and_a_second_slice() -> BoundKernel:
    # ``out[tile, :, :]`` spans the rows of ``scale`` and the reduced row;
    # no value of the kernel carries both.
    return _second_slice_meets_the_row(
        _bound(row_sums_over_a_scale_matrix, (128, 128), (128, 64))
    )


def _atomic_across_the_row_and_a_second_slice() -> BoundKernel:
    # The same region, added to with an atomic: the lowering refuses the
    # rolled row of these seeds.
    return _second_slice_meets_the_row(
        _bound(
            row_sums_added_over_a_scale_matrix,
            (128, 128),
            (128, 64),
            (128, 64, 128),
            fp32=(2,),
        )
    )


def _literal_dim_of_a_second_slice_beside_the_row() -> BoundKernel:
    # ``expand(-1, 64, -1)``: the 64 is an int, which the lowering matches
    # to block 2 by its extent.
    return _second_slice_meets_the_row(
        _bound(row_sums_of_literal_copies, (2048, 128), (2048, 64))
    )


def _value_with_the_row_and_a_second_slice() -> BoundKernel:
    return _second_slice_meets_the_row(
        _bound(row_sums_of_outer_products, (2048, 128), (2048, 64))
    )


def _two_swept_slices_in_one_value() -> BoundKernel:
    # Blocks 2 and 3, the 32-wide rows of ``a`` and the 64-wide rows of
    # ``b``, would each take the 16 threads of the 128-wide reduced row,
    # block 1, as two and four elements a thread.  Their product carries
    # both, so only one of them could index the row's axis.
    bound = _bound(row_sums_scale_outer_products, (2048, 128), (2048, 32), (2048, 64))
    spec = bound.config_spec
    assert spec.cute_static_reduction_extents == {1: 128, 2: 32, 3: 64}
    assert _reduced_blocks(bound) == {1}
    assert _loop_blocks(bound) == [1]
    device_ir = bound.host_function.device_ir
    assert not _row_is_no_run_of_packets(bound.env, device_ir, 1)
    assert slice_packet_entries(bound.env, device_ir, 1, 16) == (
        {2: 16, 3: 16},
        {2: 2, 3: 4},
    )
    assert _full_slices_meet(bound.env, device_ir)
    return bound


def _outer_tile_axis_fixed_at_128_batches() -> BoundKernel:
    # Block 0, the batch axis, is no block size of the config.  It takes 128
    # threads of its own, which would leave the 8 rows of 8 threads one
    # thread a row.
    bound = _bound(rmsnorm_heads_of_128_batches, (256, 16, 192))
    spec = bound.config_spec
    assert spec.cute_tile_fixed_block_sizes == {0: 128}
    assert [item.block_id for item in spec.block_sizes] == [1]
    return bound


def _innermost_tile_axis_fixed_at_four_heads() -> BoundKernel:
    # The rows per CTA would go to the batch axis, beside four heads each.
    bound = _bound(rmsnorm_four_heads_a_cta, (256, 16, 192))
    spec = bound.config_spec
    assert spec.cute_tile_fixed_block_sizes == {1: 4}
    assert [item.block_id for item in spec.block_sizes] == [0]
    return bound


def _nested_tile_loop_fixed_at_four_columns() -> BoundKernel:
    # Block 2, the columns of ``y``, is fixed, so the kernel's one block
    # size tiles the rows; its four columns take a thread axis beside the
    # rows per CTA and the row's threads.
    bound = _bound(row_norm_then_four_columns_a_tile, (2048, 192), (2048, 64))
    spec = bound.config_spec
    assert spec.cute_tile_fixed_block_sizes == {2: 4}
    assert [item.block_id for item in spec.block_sizes] == [0]
    assert bound.host_function.device_ir.grid_block_ids == [[0]]
    return bound


def _tile_axis_fixed_at_an_argument() -> BoundKernel:
    # One batch a CTA, but as an int argument: no size known to the seeds.
    bound = _bound(rmsnorm_heads_of_batches, (256, 16, 192), 1)
    spec = bound.config_spec
    assert spec.cute_tile_symbolic_fixed_blocks == {0}
    assert not spec.cute_tile_fixed_block_sizes
    return bound


def _slice_from_a_computed_bound() -> BoundKernel:
    # ``x[tile, n:]``: the slice starts at a node of the kernel's graph,
    # whose value the plan does not resolve, so which block it walks is
    # unknown.
    bound = _bound(row_sum_scales_computed_tail, (2048, 128))
    assert _loop_blocks(bound) == [1]
    assert _reduced_blocks(bound) == {1}
    assert bound.config_spec.cute_static_reduction_extents[1] == 128
    return bound


def _dynamic_row() -> BoundKernel:
    bound = _layernorm(2048, 192, static_shapes=False)
    assert not bound.config_spec.cute_static_reduction_extents
    return bound


def _fp64_row() -> BoundKernel:
    return _layernorm(2048, 48, dtype=torch.float64)


def _outer_tile_axis_above_its_floor() -> BoundKernel:
    # 200000 tokens of 4 heads: the token axis starts at 32 blocks.
    bound = _rotate(200000, 4, 128)
    spec = bound.config_spec
    assert spec.block_sizes[0]._fragment(spec).low == 32
    return bound


# 33 cases, one per way a kernel fails the plan: the argmax row, the pinned
# loop, the half row reduced beside the row and the stepped row twice, as
# the sub-warp rows and the floored warp rows read the plan for different
# row lengths.
@pytest.mark.parametrize(
    "declined",
    [
        _second_root_loop,
        _second_root_loop_with_a_block_size,
        _tile_loop_nested_in_the_row_loop,
        _registered_block_size_that_no_loop_tiles,
        _matmul_beside_the_row,
        _no_tile_axis,
        _loop_that_nothing_reduces,
        _loop_on_a_row_beside_the_reduced_one,
        _half_row_reduced_beside_the_row,
        _half_row_reduced_above_the_occupancy_floor,
        _rows_of_a_second_tensor_reduced,
        _no_reduction_loop,
        _argmax_row,
        _argmax_row_above_the_occupancy_floor,
        _loop_pinned_persistent,
        _loop_pinned_persistent_above_the_occupancy_floor,
        _column_reduction_on_a_square_tensor,
        _rows_of_a_strided_view,
        _row_sliced_with_a_step,
        _row_sliced_with_a_step_above_the_occupancy_floor,
        _store_across_the_row_and_a_second_slice,
        _atomic_across_the_row_and_a_second_slice,
        _literal_dim_of_a_second_slice_beside_the_row,
        _value_with_the_row_and_a_second_slice,
        _two_swept_slices_in_one_value,
        _outer_tile_axis_fixed_at_128_batches,
        _innermost_tile_axis_fixed_at_four_heads,
        _nested_tile_loop_fixed_at_four_columns,
        _tile_axis_fixed_at_an_argument,
        _slice_from_a_computed_bound,
        _dynamic_row,
        _fp64_row,
        _outer_tile_axis_above_its_floor,
    ],
    ids=lambda case: case.__name__.strip("_").replace("_", " "),
)
def test_kernels_without_a_packet_row_get_no_seed(
    declined: Callable[[], BoundKernel],
) -> None:
    bound = declined()
    for heuristic in ROW_HEURISTICS:
        assert _seeds(heuristic, bound) == []
        assert heuristic.name not in bound.config_spec.autotuner_heuristics
