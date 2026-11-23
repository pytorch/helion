"""The seeds of a transposed tile store
(``helion/_compiler/cute/autotune/heuristics/cute_transposed_store_seeds.py``,
``helion/_compiler/cute/transposed_stores.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

_NAME = "cute_transposed_store_seeds"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pad_and_transpose(
    x: torch.Tensor, padded: hl.constexpr, transpose_first: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` zero-padded to ``padded`` columns, stored transposed and as is:
    the shape of better-benchmark's ``pad_vocab_and_transpose``, which
    ``sum_6119d597d188`` stores transposed first and ``sum_81b4fd73f8d1``
    padded first (each store has the other's tile contiguous)."""
    rows, cols = x.shape
    out_t = torch.empty([padded, rows], dtype=x.dtype, device=x.device)
    out = torch.empty([rows, padded], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, padded]):
        in_range = tile_c.index < cols
        value = hl.load(x, [tile_r, tile_c], extra_mask=in_range[None, :])
        value = torch.where(in_range[None, :], value, 0.0)
        if transpose_first:
            out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
        if not transpose_first:
            out_t[tile_c, tile_r] = value.permute(1, 0)
    return out_t, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_transpose(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_input_copy(xt: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with the load transposed against a plain store."""
    cols, rows = xt.shape
    out = torch.empty([rows, cols], dtype=xt.dtype, device=xt.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = (xt[tile_c, tile_r] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _wide_column_transpose(x: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with column tiles of 2048 or more."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    block_c = hl.register_block_size(2048, 4096)
    for tile_r, tile_c in hl.tile([rows, cols], block_size=[None, block_c]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_transpose(x: torch.Tensor) -> torch.Tensor:
    """``_scaled_transpose`` with the column tiles in a loop of their own."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        for tile_c in hl.tile(cols):
            out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_beside_a_row_loop(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_scaled_transpose`` and a loop over ``y`` of a tile of its own."""
    rows, cols = x.shape
    out = torch.empty([cols, rows], dtype=x.dtype, device=x.device)
    out_y = torch.empty_like(y)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_c, tile_r] = (x[tile_r, tile_c] * 2).T
    for tile_n in hl.tile(y.size(0)):
        out_y[tile_n] = y[tile_n] + 1
    return out, out_y


def _cast_and_transpose(x: torch.Tensor, out_t: torch.Tensor) -> torch.Tensor:
    """``x`` cast to ``out_t``'s dtype and stored transposed into ``out_t`` and
    as is: the shape of better-benchmark's ``cast_and_transpose``
    (``pointwise_52fe1925ccda``), whose ``out_t`` is the first columns of a
    padded buffer."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


_cast_both_ways = helion.kernel(
    _cast_and_transpose, backend="cute", static_shapes=True, autotune_effort="none"
)
_cast_both_ways_dynamic = helion.kernel(
    _cast_and_transpose, backend="cute", static_shapes=False, autotune_effort="none"
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_cast_both_ways(
    x: torch.Tensor, out_t: torch.Tensor, lane: hl.constexpr
) -> torch.Tensor:
    """``_cast_both_ways`` of ``x`` one column over (``lane``: the load's
    subscript on its contiguous dim is the tile's index plus a scalar) or
    one row down."""
    rows, cols = out_t.shape[1], out_t.shape[0]
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        if lane:
            value = x[tile_r, tile_c.index + 1]
        else:
            value = x[tile_r.index + 1, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_both_ways(
    x: torch.Tensor, y: torch.Tensor, out_t: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` of ``x + y``: two loads of the tile."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c] + y[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_of_both_layouts(
    x: torch.Tensor, yt: torch.Tensor, out_t: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` of ``x`` plus the transposed ``yt``, a load that
    has the store's tile on its contiguous dim."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c] + yt[tile_c, tile_r].permute(1, 0)
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_window_both_ways(
    x: torch.Tensor, out_t: torch.Tensor, begin: hl.constexpr, end: hl.constexpr
) -> torch.Tensor:
    """``_cast_both_ways`` of the columns from ``begin`` to ``end``."""
    rows = x.size(0)
    out = torch.zeros_like(x, dtype=out_t.dtype)
    for tile_r, tile_c in hl.tile([0, begin], [rows, end]):
        value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_if(
    x: torch.Tensor, out_t: torch.Tensor, flag: torch.Tensor, predicate: hl.constexpr
) -> torch.Tensor:
    """``_cast_both_ways`` under a branch, a graph of its own: on a tensor's
    value, on a block size (which codegen folds away) or on a tile's
    begin."""
    rows, cols = x.shape
    out = torch.zeros([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        if predicate == "tensor":
            taken = flag[0] > 0
        elif predicate == "block size":
            taken = tile_c.block_size > 32
        else:
            taken = tile_r.begin > 0
        if taken:
            value = x[tile_r, tile_c]
            out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
            out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_of_a_branch_load(
    x: torch.Tensor, out_t: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` with the load alone under a branch on a tensor's
    value, the two stores after it."""
    rows, cols = x.shape
    out = torch.zeros([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = hl.zeros([tile_r, tile_c], dtype=x.dtype)
        if flag[0] > 0:
            value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_into_both(
    x: torch.Tensor, out_t: torch.Tensor, out: torch.Tensor, masked: hl.constexpr
) -> torch.Tensor:
    """``_cast_both_ways`` into an ``out`` that is an argument, stored under
    an extra mask if ``masked``."""
    rows, cols = x.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        if masked:
            hl.store(out, [tile_r, tile_c], value.to(out.dtype), extra_mask=value > 0)
        else:
            out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_into_a_branch(
    x: torch.Tensor, out_t: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` with the two stores alone under a branch on a
    tensor's value, the load before it."""
    rows, cols = x.shape
    out = torch.zeros([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c]
        if flag[0] > 0:
            out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
            out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_beside_a_copy(
    x: torch.Tensor,
    y: torch.Tensor,
    out_t: torch.Tensor,
    flag: torch.Tensor,
    branch: hl.constexpr,
) -> torch.Tensor:
    """A copy of ``x`` beside a transposed cast of ``y``, which sits under a
    branch on a tensor's value (``branch``) or beside the copy."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = x[tile_r, tile_c].to(out.dtype)
        if branch:
            if flag[0] > 0:
                out_t[tile_c, tile_r] = y[tile_r, tile_c].permute(1, 0).to(out_t.dtype)
        else:
            out_t[tile_c, tile_r] = y[tile_r, tile_c].permute(1, 0).to(out_t.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_beside_a_reduction(
    x: torch.Tensor, y: torch.Tensor, out_t: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` of ``x`` plus the row sums of ``y``: a reduction
    dim beside the two tiles."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c] + y[tile_r, :].sum(-1)[:, None]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_beside_row_sums(x: torch.Tensor, out_t: torch.Tensor) -> torch.Tensor:
    """A transposed cast of ``x`` beside its row sums, a reduction of the
    tile's lanes."""
    rows, cols = x.shape
    sums = torch.zeros([rows], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        hl.atomic_add(sums, [tile_r], value.sum(-1))
    return sums


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_beside_rows(
    x: torch.Tensor, y: torch.Tensor, out_t: torch.Tensor
) -> torch.Tensor:
    """``_cast_both_ways`` and a copy of whole rows of ``y``, whose untiled
    dim takes threads."""
    rows, cols = x.shape
    out = torch.empty_like(y)
    for tile_r, tile_c in hl.tile([rows, cols]):
        out_t[tile_c, tile_r] = x[tile_r, tile_c].permute(1, 0).to(out_t.dtype)
        out[tile_r, :] = y[tile_r, :] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_in_place(x: torch.Tensor, flip: hl.constexpr) -> torch.Tensor:
    """A square ``x`` stored transposed into itself (``flip``) or, with its
    sign flipped, as it was loaded, beside a transposed copy."""
    n = x.size(0)
    out_t = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([n, n]):
        value = x[tile_r, tile_c]
        if flip:
            x[tile_c, tile_r] = value.permute(1, 0)
        else:
            out_t[tile_c, tile_r] = value.permute(1, 0)
            x[tile_r, tile_c] = -value
    return out_t


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_beside_a_flip(
    x: torch.Tensor, out_t: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """A transposed cast of a square ``x`` beside ``z`` stored transposed
    into itself, read with the store's tile contiguous: not a load the
    first store transposes."""
    n = x.size(0)
    for tile_r, tile_c in hl.tile([n, n]):
        out_t[tile_c, tile_r] = x[tile_r, tile_c].permute(1, 0).to(out_t.dtype)
        z[tile_r, tile_c] = z[tile_c, tile_r].permute(1, 0)
    return out_t


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _index_sums_both_ways(out_t: torch.Tensor) -> torch.Tensor:
    """A tile of index sums stored transposed and as is: each store is the
    other's witness, and nothing is loaded."""
    rows, cols = out_t.shape[1], out_t.shape[0]
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=out_t.device)
    for tile_r, tile_c in hl.tile([rows, cols]):
        value = (tile_r.index[:, None] + tile_c.index[None, :]).to(out.dtype)
        out_t[tile_c, tile_r] = value.permute(1, 0)
        out[tile_r, tile_c] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _cast_both_ways_in_blocks(
    x: torch.Tensor, out_t: torch.Tensor, rows_low: hl.constexpr, cols_low: hl.constexpr
) -> torch.Tensor:
    """``_cast_both_ways`` with registered tiles: the smallest block of the
    rows is ``rows_low``, that of the columns ``cols_low``."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=out_t.dtype, device=x.device)
    block_r = hl.register_block_size(rows_low, 2 * rows_low)
    block_c = hl.register_block_size(cols_low, 2 * cols_low)
    for tile_r, tile_c in hl.tile([rows, cols], block_size=[block_r, block_c]):
        value = x[tile_r, tile_c]
        out_t[tile_c, tile_r] = value.permute(1, 0).to(out_t.dtype)
        out[tile_r, tile_c] = value.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.shape):
        out[tile_r, tile_c] = x[tile_r, tile_c] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_rows(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.shape):
        out[tile_r, tile_c] = x[tile_r, tile_c] * scale[tile_r][:, None]
    return out


def _to_channels_last(x: torch.Tensor) -> torch.Tensor:
    """``[N, C, S]`` copied into the channels-last ``[N, S, C]``: the layout
    copy of better-benchmark's ``pointwise_06377502b8f5``."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


_channels_last = helion.kernel(
    _to_channels_last, backend="cute", static_shapes=True, autotune_effort="none"
)
_channels_last_dynamic = helion.kernel(
    _to_channels_last, backend="cute", static_shapes=False, autotune_effort="none"
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last(x: torch.Tensor) -> torch.Tensor:
    """The reverse copy: channels-last ``[N, S, C]`` into ``[N, C, S]``."""
    n, plane, channels = x.shape
    out = torch.empty([n, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_s, tile_c].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` into a tensor of the caller."""
    n, channels, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_if(
    x: torch.Tensor, flag: torch.Tensor, predicate: hl.constexpr
) -> torch.Tensor:
    """``_to_channels_last`` under a branch, a graph of its own: on a
    tensor's value, on a block size (which codegen folds away) or on a
    tile's begin."""
    n, channels, plane = x.shape
    out = torch.zeros([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        if predicate == "tensor":
            taken = flag[0] > 0
        elif predicate == "block size":
            taken = tile_c.block_size > 4
        else:
            taken = tile_s.begin > 0
        if taken:
            out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_beside_a_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` and a loop over ``y`` of a tile of its own."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    out_y = torch.empty_like(y)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    for tile_k in hl.tile(y.size(0)):
        out_y[tile_k] = y[tile_k] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_beside_rows(
    x: torch.Tensor, y: torch.Tensor, reduce: hl.constexpr
) -> torch.Tensor:
    """``_to_channels_last`` beside whole rows of ``y``, whose untiled dim
    takes threads: a copy of them, or (``reduce``) their sums."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    out_y = torch.empty_like(y)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
        if reduce:
            value = value + y[tile_n, :].sum(-1)[:, None, None].to(x.dtype)
        else:
            out_y[tile_n, :] = y[tile_n, :] + 1
        out[tile_n, tile_s, tile_c] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_and_channel_sums(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` beside the sums over the positions, a reduction
    of the tile's lanes."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    sums = torch.zeros([n, channels], dtype=torch.float32, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s]
        out[tile_n, tile_s, tile_c] = value.permute(0, 2, 1)
        hl.atomic_add(sums, [tile_n, tile_c], value.to(torch.float32).sum(-1))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_of_a_sum(
    x: torch.Tensor, y: torch.Tensor, into_y: hl.constexpr
) -> torch.Tensor:
    """``x + y`` of as many channels as positions stored channels-last, and
    (``into_y``) into the second load's tensor as well."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = (x[tile_n, tile_c, tile_s] + y[tile_n, tile_c, tile_s]).permute(0, 2, 1)
        out[tile_n, tile_s, tile_c] = value
        if into_y:
            y[tile_n, tile_s, tile_c] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_and_an_atomic(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` beside an atomic add of the whole tile."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    acc = torch.zeros([n, channels, plane], dtype=torch.float32, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s]
        out[tile_n, tile_s, tile_c] = value.permute(0, 2, 1)
        hl.atomic_add(acc, [tile_n, tile_c, tile_s], value.to(torch.float32))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_in_place(x: torch.Tensor, flip: hl.constexpr) -> torch.Tensor:
    """``x`` of as many channels as positions stored channels-last into
    itself (``flip``) or, with its sign flipped, as it was loaded, beside a
    channels-last copy."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s]
        if flip:
            x[tile_n, tile_s, tile_c] = value.permute(0, 2, 1)
        else:
            out[tile_n, tile_s, tile_c] = value.permute(0, 2, 1)
            x[tile_n, tile_c, tile_s] = -value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_masked(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with an extra mask on the store."""
    n, channels, plane = x.shape
    out = torch.zeros([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
        hl.store(out, [tile_n, tile_s, tile_c], value, extra_mask=value > 0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_masked_load(
    x: torch.Tensor, keep: hl.constexpr
) -> torch.Tensor:
    """``_from_channels_last`` with an extra mask on the load: the first
    ``keep`` channels, zeros for the others."""
    n, plane, channels = x.shape
    out = torch.empty([n, channels, plane], dtype=x.dtype, device=x.device)
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        kept = (tile_c.index < keep)[None, None, :]
        value = hl.load(x, [tile_n, tile_s, tile_c], extra_mask=kept)
        out[tile_n, tile_c, tile_s] = value.permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_per_channel_tile(x: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with the batch and position tiles in a loop of
    their own under the channel tiles."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for tile_c in hl.tile(channels):
        for tile_n, tile_s in hl.tile([n, plane]):
            out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_under_a_loop(x: torch.Tensor, rounds: torch.Tensor) -> torch.Tensor:
    """``_to_channels_last`` with its three tiles in a loop under another
    tile's (one copy per tile of ``rounds``)."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    for _tile_r in hl.tile(rounds.size(0)):
        for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
            out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_window(
    x: torch.Tensor, out: torch.Tensor, begin: hl.constexpr, end: hl.constexpr
) -> torch.Tensor:
    """``_channels_last_into`` for the channels from ``begin`` to ``end``."""
    n, _, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([0, begin, 0], [n, end, plane]):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_window(
    x: torch.Tensor, out: torch.Tensor, begin: hl.constexpr, end: hl.constexpr
) -> torch.Tensor:
    """The reverse copy for the channels from ``begin`` to ``end``."""
    n, plane, _ = x.shape
    for tile_n, tile_s, tile_c in hl.tile([0, 0, begin], [n, plane, end]):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_s, tile_c].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_shifted_store(
    x: torch.Tensor, out: torch.Tensor, lane: hl.constexpr
) -> torch.Tensor:
    """``_channels_last_into`` through a store subscript that adds a scalar
    to a tile's index: 16 channels into the rows of ``out`` (``lane``), or
    two rows down."""
    n, channels, plane = x.shape
    for tile_n, tile_c, tile_s in hl.tile([n, channels, plane]):
        value = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
        if lane:
            out[tile_n, tile_s, tile_c.index + 16] = value
        else:
            out[tile_n, tile_s.index + 2, tile_c] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _from_channels_last_shifted_load(
    x: torch.Tensor, out: torch.Tensor, lane: hl.constexpr
) -> torch.Tensor:
    """The reverse copy through a load subscript that adds a scalar to a
    tile's index: of the channels of ``x`` from 16 on (``lane``), or of its
    rows from two down."""
    n, channels, plane = out.shape
    for tile_n, tile_s, tile_c in hl.tile([n, plane, channels]):
        if lane:
            value = x[tile_n, tile_s, tile_c.index + 16]
        else:
            value = x[tile_n, tile_s.index + 2, tile_c]
        out[tile_n, tile_c, tile_s] = value.permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channels_last_wide_channel_tiles(
    x: torch.Tensor, smallest: hl.constexpr
) -> torch.Tensor:
    """``_to_channels_last`` with channel tiles of ``smallest`` or more."""
    n, channels, plane = x.shape
    out = torch.empty([n, plane, channels], dtype=x.dtype, device=x.device)
    block_c = hl.register_block_size(smallest, 2048)
    for tile_n, tile_c, tile_s in hl.tile(
        [n, channels, plane], block_size=[None, block_c, None]
    ):
        out[tile_n, tile_s, tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """The same layout change over four tiles."""
    n, channels, height, width = x.shape
    out = torch.empty([n, height, width, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c, tile_h, tile_w in hl.tile([n, channels, height, width]):
        out[tile_n, tile_h, tile_w, tile_c] = x[tile_n, tile_c, tile_h, tile_w].permute(
            0, 2, 3, 1
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_scale(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_n, tile_c, tile_s in hl.tile(x.shape):
        out[tile_n, tile_c, tile_s] = x[tile_n, tile_c, tile_s] * 2
    return out


def _batched_seeds(bound: BoundKernel) -> list[dict[str, object]]:
    """The seeds of a batched layout change: the only ones that lead the
    loop order with the second tile and pin the default store policy."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_store_policy") == "default"
        and "loop_orders" in seed.config
    ]


def _seeds(bound: BoundKernel, packet: int) -> list[dict[str, object]]:
    """The seeds with 64 threads along the columns, two along the rows and
    ``packet``-row vectors."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("num_threads") == [2, 64]
        and seed.config.get("cute_vector_widths") == [packet, 1]
    ]


def _load_packet_seeds(bound: BoundKernel) -> list[dict[str, object]]:
    """The seeds with the packets on the load's tile: the only ones that
    name a loop order without a lane unroll (the store-packet seeds), a
    store policy (the batched ones) or a CTAs-per-SM minimum (the wide-row
    seeds of the pointwise heuristic)."""
    return [
        dict(seed.config)
        for seed in bound.config_spec.compiler_seed_configs
        if "loop_orders" in seed.config
        and not {"cute_lane_unroll", "cute_store_policy", "cute_min_blocks_per_mp"}
        & set(seed.config)
    ]


def _padded_rows(cols: int, rows: int, dtype: torch.dtype) -> torch.Tensor:
    """A ``[cols, rows]`` view of a buffer whose rows are padded to a
    multiple of 16 elements, as the port's ``out_t`` is."""
    return torch.empty(cols, rows + (-rows) % 16, dtype=dtype)[:, :rows]


@pytest.mark.parametrize("transpose_first", [True, False])
def test_bf16_seeds_give_the_threads_packets_of_rows(transpose_first: bool) -> None:
    """With either store first.  Stored first, the padded copy is transposed
    against ``out_t`` as well; the seeds follow ``out_t``, the store the load
    disagrees with, and do not run the threads along the rows, where every
    warp load would read rows apart."""
    args = (torch.empty(4096, 8189, dtype=torch.bfloat16), 8192, transpose_first)
    with cpu_render_target():
        bound = cpu_bind(_pad_and_transpose, args)
        assert _NAME in bound.config_spec.autotuner_heuristics
        seeds = _seeds(bound, 16)
        # 32-byte packets of the rows, two threads per column, 64 threads
        # along the columns; four and two column lanes per thread.
        assert [seed["block_sizes"] for seed in seeds] == [[32, 256], [64, 128]]
        assert [seed["num_threads"] for seed in seeds] == [[2, 64], [2, 64]]
        assert [seed["cute_vector_widths"] for seed in seeds] == [[16, 1], [16, 1]]
        assert [seed["cute_lane_unroll"] for seed in seeds] == [4, 2]
        # Thread axis 0 along the columns: their tile leads the loop order,
        # and both tiles' lanes are strided.
        assert [seed["loop_orders"] for seed in seeds] == [[[1, 0]], [[1, 0]]]
        assert [seed["cute_lane_layouts"] for seed in seeds] == [["strided"] * 2] * 2
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    # Every lane's loads issue before the stores.
    assert "_lane_unroll_loads_0" in code


def test_a_transposed_load_runs_the_threads_along_the_store_rows() -> None:
    """The store's other dim is its rows here: 64 threads along them, which
    lead the loop order, and the packets along its columns."""
    with cpu_render_target():
        bound = cpu_bind(
            _transposed_input_copy, (torch.empty(4096, 2048, dtype=torch.bfloat16),)
        )
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("num_threads") == [64, 2]
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[256, 32], [128, 64]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 16], [1, 16]]
    assert [seed["loop_orders"] for seed in seeds] == [[[0, 1]], [[0, 1]]]
    assert [seed["cute_lane_layouts"] for seed in seeds] == [["strided"] * 2] * 2


def test_fp32_seeds_hold_eight_rows_per_packet() -> None:
    with cpu_render_target():
        bound = cpu_bind(_scaled_transpose, (torch.randn(2048, 4096),))
        seeds = _seeds(bound, 8)
    assert [seed["block_sizes"] for seed in seeds] == [[16, 256], [32, 128]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [[8, 1], [8, 1]]


def test_one_byte_seeds_take_the_widest_vector_the_spec_admits() -> None:
    # 32 one-byte rows would fill the 32-byte packet, but no vector width
    # reaches 32 elements: the seeds take 16, which the autotuner's transfer
    # keeps (a width of 32 would be an invalid config, dropped there).
    x = torch.randint(-8, 8, (2048, 4096), dtype=torch.int8)
    with cpu_render_target():
        bound = cpu_bind(_scaled_transpose, (x,))
        seeds = _seeds(bound, 16)
        generation = bound.config_spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[64, 256], [128, 128]]
    for seed in seeds:
        assert any(seed.items() <= config.items() for config in transferred)


def test_a_wide_column_tile_keeps_the_largest_lane_unroll() -> None:
    """Column tiles of 2048 over 64 threads hold 32 lanes per thread, past
    the largest ``cute_lane_unroll`` factor: the seeds take 16, which the
    autotuner's transfer keeps (it turned 32 into 1, losing the
    loads-first lanes)."""
    with cpu_render_target():
        bound = cpu_bind(_wide_column_transpose, (torch.empty(1024, 8192),))
        # The registered block comes first in the config's lists.
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("num_threads") == [64, 2]
        ]
        generation = bound.config_spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[2048, 16], [2048, 32]]
    assert [seed["cute_lane_unroll"] for seed in seeds] == [16, 16]
    for seed in seeds:
        assert any(seed.items() <= config.items() for config in transferred)


def test_another_loops_tile_keeps_its_default_size() -> None:
    # The seeds size the transposed pair; the other loop's tile takes its
    # default (a seed with that slot left out was dropped).
    with cpu_render_target():
        bound = cpu_bind(
            _transpose_beside_a_row_loop,
            (torch.empty(2048, 4096), torch.empty(1000)),
        )
        default = bound.config_spec.default_config().config["block_sizes"]
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_lane_layouts") == ["strided", "strided", "blocked"]
        ]
    assert [seed["block_sizes"] for seed in seeds] == [
        [16, 256, default[2]],
        [32, 128, default[2]],
    ]


def test_one_column_lane_per_thread_unrolls_nothing() -> None:
    # 32 columns over 32 threads: one column lane each, nothing to unroll.
    with cpu_render_target():
        bound = cpu_bind(
            _scaled_transpose, (torch.empty(256, 32, dtype=torch.bfloat16),)
        )
        seeds = [
            dict(seed.config)
            for seed in bound.config_spec.compiler_seed_configs
            if seed.config.get("cute_lane_layouts") == ["strided"] * 2
        ]
    assert [seed["block_sizes"] for seed in seeds] == [[32, 32], [64, 32]]
    assert [seed["cute_lane_unroll"] for seed in seeds] == [1, 1]


def test_no_seeds_for_tiles_of_nested_loops() -> None:
    # No loop order spans both tiles, so the seeds could neither lead with
    # the column dim nor unroll lanes the two loops share.
    with cpu_render_target():
        bound = cpu_bind(_nested_transpose, (torch.empty(256, 512),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_packet_seeds_follow_the_branches_codegen_folds() -> None:
    """A branch is a graph of its own.  One on a block size is folded away
    when the code is generated, and the accesses under it keep their
    packets and their seeds; under one on a tensor's value or on a tile's
    begin the lowering loads and stores a tile's lanes one by one, and the
    seeds planted on a packet proof, the load-packet seed and the batched
    ones, are not planted.  (The store-packet seeds of a rank-2 store,
    which prove no packets, are planted under every branch.)"""
    bf16 = torch.bfloat16
    out_t = _padded_rows(256, 1001, bf16)
    x = torch.empty(1001, 256)
    wide = torch.empty(4, 96, 48, dtype=bf16)
    flag = torch.ones(1)
    with cpu_render_target():
        bound = cpu_bind(_cast_both_ways_if, (x, out_t, flag, "block size"))
        (seed,) = _load_packet_seeds(bound)
        code = bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
        assert "block=(2, 32, 1)" in code
        assert code.count("cute.arch.load(") == 1
        bound = cpu_bind(_channels_last_if, (wide, flag, "block size"))
        seeds = _batched_seeds(bound)
        assert len(seeds) == 6
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        for predicate in ("tensor", "begin"):
            bound = cpu_bind(_cast_both_ways_if, (x, out_t, flag, predicate))
            assert len(_seeds(bound, 16)) == 2
            assert _load_packet_seeds(bound) == []
            bound = cpu_bind(_channels_last_if, (wide, flag, predicate))
            assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_seeds_without_a_transposed_store() -> None:
    with cpu_render_target():
        bound = cpu_bind(_copy, (torch.randn(256, 512),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_seeds_beside_a_row_broadcast() -> None:
    # ``scale`` has the store's rows on its contiguous dim, but it indexes
    # only the rows; an expanded column (strides (1, 0)) has them contiguous
    # too, and its other dim does not move: nothing in either kernel is
    # transposed.
    with cpu_render_target():
        for bound in (
            cpu_bind(_scale_rows, (torch.empty(256, 512), torch.empty(256))),
            cpu_bind(_copy, (torch.empty(256, 1).expand(256, 512),)),
        ):
            assert _NAME not in bound.config_spec.autotuner_heuristics


def test_a_load_that_takes_packets_seeds_two_packet_threads() -> None:
    """The shape of ``pointwise_52fe1925ccda``, small: fp32 ``[1001, 256]``
    cast to bf16 and stored as is and transposed, into rows of 1001 elements
    that no packet divides.  The load takes 32-byte packets of its rows, so
    one seed has two threads along them on thread axis 0 (their tile leads
    the loop order), each with four packets, the two threads' packets
    alternating (strided lanes), and 32 threads along the transposed rows.
    A bf16 load has 16 elements to the packet and tiles twice as long.  The
    other seeds keep their places: a twin whose load is one element into its
    storage, which no packet survives, has the same seeds but this one."""
    bf16 = torch.bfloat16
    out_t = _padded_rows(256, 1001, bf16)
    with cpu_render_target():
        bound = cpu_bind(_cast_both_ways, (torch.empty(1001, 256), out_t))
        (seed,) = _load_packet_seeds(bound)
        assert seed == {
            "block_sizes": [32, 64],
            "num_threads": [32, 2],
            "cute_vector_widths": [1, 8],
            "cute_lane_layouts": ["blocked", "strided"],
            "loop_orders": [[1, 0]],
        }
        code = bound.to_code(bound.config_spec.normalized_config(helion.Config(**seed)))
        # Two packet threads on thread axis 0, 32 rows on axis 1; the load
        # and the untransposed store take packets.
        assert "block=(2, 32, 1)" in code
        assert code.count("cute.arch.load(") == 1
        assert code.count("_cute_store_u16_vec(") == 1

        seeds = [dict(s.config) for s in bound.config_spec.compiler_seed_configs]
        offset = torch.empty(1001 * 256 + 1)[1:].view(1001, 256)
        twin = cpu_bind(_cast_both_ways, (offset, out_t))
        assert _load_packet_seeds(twin) == []
        assert [s for s in seeds if s != seed] == [
            dict(s.config) for s in twin.config_spec.compiler_seed_configs
        ]

        bound = cpu_bind(_cast_both_ways, (torch.empty(1001, 256, dtype=bf16), out_t))
        (seed,) = _load_packet_seeds(bound)
        assert seed["block_sizes"] == [32, 128]
        assert seed["num_threads"] == [32, 2]
        assert seed["cute_vector_widths"] == [1, 16]


def test_a_load_packet_seed_beside_the_stores_packet_seeds() -> None:
    """Where the transposed rows take packets too (1024 elements), the
    store's seeds stay first and this one follows.  A store that only
    transposes gets it as well, and a transposed load beside a plain store
    has the packet threads along its rows, which are the store's columns."""
    with cpu_render_target():
        bound = cpu_bind(
            _cast_both_ways,
            (torch.empty(1024, 256), torch.empty(256, 1024, dtype=torch.bfloat16)),
        )
        seeds = [dict(s.config) for s in bound.config_spec.compiler_seed_configs]
        (seed,) = _load_packet_seeds(bound)
        store_seeds = _seeds(bound, 16)
        assert len(store_seeds) == 2
        assert [seeds.index(s) for s in store_seeds] == [
            seeds.index(seed) - 2,
            seeds.index(seed) - 1,
        ]

        bound = cpu_bind(_scaled_transpose, (torch.randn(2048, 4096),))
        (seed,) = _load_packet_seeds(bound)
        assert seed["block_sizes"] == [32, 64]
        assert seed["num_threads"] == [32, 2]

        bound = cpu_bind(
            _transposed_input_copy, (torch.empty(4096, 2048, dtype=torch.bfloat16),)
        )
        (seed,) = _load_packet_seeds(bound)
        assert seed == {
            "block_sizes": [128, 32],
            "num_threads": [2, 32],
            "cute_vector_widths": [16, 1],
            "cute_lane_layouts": ["strided", "blocked"],
            "loop_orders": [[0, 1]],
        }


def test_larger_smallest_blocks_take_more_packets_and_rows() -> None:
    """The seed's tiles are the search's: a smallest block of 256 columns
    gives each of the two packet threads 16 packets, one of 64 or 256 rows
    that many threads along them, and one of 512 rows, which would be 1024
    threads, no seed."""
    x = torch.empty(2048, 2048)
    out_t = torch.empty(2048, 2048, dtype=torch.bfloat16)
    with cpu_render_target():
        for rows_low, cols_low, sizes in (
            (64, 256, [64, 256]),
            (256, 64, [256, 64]),
            (512, 64, None),
        ):
            bound = cpu_bind(_cast_both_ways_in_blocks, (x, out_t, rows_low, cols_low))
            seeds = _load_packet_seeds(bound)
            if sizes is None:
                assert seeds == []
                continue
            (seed,) = seeds
            assert seed["block_sizes"] == sizes
            assert seed["num_threads"] == [sizes[0], 2]
            assert seed["cute_vector_widths"] == [1, 8]


def test_no_load_packet_seed_for_tiles_smaller_than_the_seeds() -> None:
    """The largest block of a tile is its extent rounded up to a power of
    two: 32 fp32 columns hold four of the seed's eight packets, 64 bf16
    columns likewise, and 16 rows half its threads.  40 columns and 17 rows
    get the seed's blocks, as partial tiles."""
    bf16 = torch.bfloat16
    with cpu_render_target():
        for x in (
            torch.empty(1001, 32),
            torch.empty(1001, 64, dtype=bf16),
            torch.empty(16, 256),
        ):
            rows, cols = x.shape
            bound = cpu_bind(_cast_both_ways, (x, _padded_rows(cols, rows, bf16)))
            assert _NAME in bound.config_spec.autotuner_heuristics
            assert _load_packet_seeds(bound) == []
        bound = cpu_bind(
            _cast_both_ways, (torch.empty(17, 40), _padded_rows(40, 17, bf16))
        )
        (seed,) = _load_packet_seeds(bound)
        assert seed["block_sizes"] == [32, 64]


def test_load_packet_seeds_need_packets_the_lowering_proves() -> None:
    """The conditions of the batched seeds' packets, for the loads a rank-2
    store transposes; the heuristic's other seeds do not depend on them.
    Every refusal has a twin beside it that differs in the one condition and
    gets the seed."""
    bf16 = torch.bfloat16
    out_t = _padded_rows(256, 1001, bf16)
    x = torch.empty(1001, 256)
    offset = torch.empty(1001 * 256 + 1)[1:].view(1001, 256)
    with cpu_render_target():
        for kernel, args in (
            # Packets are bf16, fp16 and fp32.
            (
                _cast_both_ways,
                (x.to(torch.int32), _padded_rows(256, 1001, torch.int32)),
            ),
            (_cast_both_ways, (x.double(), out_t)),
            # Rows of 250 elements, which end inside a packet, as does a
            # tile of the first 252 of 256.
            (_cast_both_ways, (torch.empty(1001, 250), _padded_rows(250, 1001, bf16))),
            (_cast_window_both_ways, (x, out_t, 0, 252)),
            # Tiles that begin four elements into a packet.
            (_cast_window_both_ways, (x, out_t, 4, 252)),
            # A base one element into its storage, and rows 260 elements
            # apart: packets off their alignment.
            (_cast_both_ways, (offset, out_t)),
            (_cast_both_ways, (torch.empty(1001, 260)[:, :256], out_t)),
            # A mask is not proven uniform over a packet, nor a scalar added
            # to the tile's index along the rows a multiple of it.
            (_pad_and_transpose, (x, 256, True)),
            (_shifted_cast_both_ways, (torch.empty(1001, 264), out_t, True)),
            # No static extent to prove the packets on.
            (_cast_both_ways_dynamic, (x, out_t)),
            # A tile stored both ways that nothing loads.
            (_index_sums_both_ways, (_padded_rows(256, 1001, torch.float32),)),
            # The store of the tile the other way round takes no packets: a
            # base one element into its storage, an extra mask.
            (
                _cast_into_both,
                (
                    x,
                    out_t,
                    torch.empty(1001 * 256 + 1, dtype=bf16)[1:].view(1001, 256),
                    False,
                ),
            ),
            (_cast_into_both, (x, out_t, torch.empty(1001, 256, dtype=bf16), True)),
            # An access of the two tiles under a branch on a tensor's value,
            # a graph of its own, where the lowering takes no packets: the
            # load, the stores, or the transposed store and its load beside
            # a copy whose load sits in the loop.
            (_cast_both_ways_of_a_branch_load, (x, out_t, torch.ones(1))),
            (_cast_both_ways_into_a_branch, (x, out_t, torch.ones(1))),
            (_transpose_beside_a_copy, (x, x.clone(), out_t, torch.ones(1), True)),
            # Not pointwise: beside a reduction of the tile's lanes the
            # seed's config does not render.
            (_transpose_beside_row_sums, (x, out_t)),
            # The CTA is not the pair's alone: the threads of a reduction
            # dim or of the untiled dim of whole rows take thread axis 0,
            # and another root loop's the place of the two packet threads.
            (_cast_both_ways_beside_a_reduction, (x, torch.empty(1001, 64), out_t)),
            (_cast_both_ways_beside_rows, (x, torch.empty(1001, 8), out_t)),
            (
                _transpose_beside_a_row_loop,
                (torch.empty(2048, 4096), torch.empty(1000)),
            ),
            # A store into a tensor the kernel loads: transposed into
            # itself, which the code generator refuses, also for a tensor
            # the store does not transpose; and as it was loaded, which
            # renders the layout and is declined on purpose (no in-place
            # kernel was measured).
            (_transpose_in_place, (torch.empty(1024, 1024), True)),
            (
                _transpose_beside_a_flip,
                (
                    torch.empty(1024, 1024),
                    torch.empty(1024, 1024, dtype=bf16),
                    torch.empty(1024, 1024),
                ),
            ),
            (_transpose_in_place, (torch.empty(1024, 1024), False)),
            # Two loads: packets of 8 and of 16 elements, and one load that
            # takes none.
            (_sum_both_ways, (x, x.to(bf16), out_t)),
            (_sum_both_ways, (x, offset, out_t)),
        ):
            bound = cpu_bind(kernel, args)
            assert _NAME in bound.config_spec.autotuner_heuristics
            assert _load_packet_seeds(bound) == []

        # Before SM100 a packet is 16 bytes at most.
        bound = cpu_bind(_cast_both_ways, (x, out_t), capability=(9, 0))
        assert _load_packet_seeds(bound) == []

        for kernel, args in (
            (_cast_into_both, (x, out_t, torch.empty(1001, 256, dtype=bf16), False)),
            (_cast_window_both_ways, (x, out_t, 8, 248)),
            (_cast_both_ways, (torch.empty(1001, 264)[:, :256], out_t)),
            # A scalar added to the index of the other tile moves no packet.
            (_shifted_cast_both_ways, (torch.empty(1002, 256), out_t, False)),
            (_sum_both_ways, (x, x, out_t)),
            (_transpose_beside_a_copy, (x, x.clone(), out_t, torch.ones(1), False)),
            # A load with the store's tile contiguous is not one the store
            # transposes: its packets are not asked for.
            (_sum_of_both_layouts, (x, torch.empty(256, 1001), out_t)),
        ):
            bound = cpu_bind(kernel, args)
            (seed,) = _load_packet_seeds(bound)
            assert seed["cute_vector_widths"] == [1, 8]
            # The layout the seed names: two packet threads, packet loads.
            code = bound.to_code(
                bound.config_spec.normalized_config(helion.Config(**seed))
            )
            assert "block=(2, 32, 1)" in code
            assert "cute.arch.load(" in code


def test_the_load_packet_seed_leaves_the_search_space_as_it_was() -> None:
    """The seed opens no knob: a load it declines (one element into its
    storage) has the search space of one it serves."""
    out_t = _padded_rows(256, 1001, torch.bfloat16)
    with cpu_render_target():
        served = cpu_bind(_cast_both_ways, (torch.empty(1001, 256), out_t))
        declined = cpu_bind(
            _cast_both_ways, (torch.empty(1001 * 256 + 1)[1:].view(1001, 256), out_t)
        )
        assert len(_load_packet_seeds(served)) == 1
        assert _load_packet_seeds(declined) == []
        assert (
            served.config_spec.structural_fingerprint()
            == declined.config_spec.structural_fingerprint()
        )


def test_a_batched_layout_change_stores_whole_sectors() -> None:
    """NCHW to channels-last: the store has the channels contiguous, the load
    the positions.  Every thread stores one packet of channels and loads it
    at one position; the channel threads lead the thread order.  (Rows of 50
    positions hold no packets of the load.)"""
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last, (torch.empty(4, 96, 50, dtype=torch.bfloat16),)
        )
        spec = bound.config_spec
        assert _NAME in spec.autotuner_heuristics
        seeds = _batched_seeds(bound)
        # 32-byte packets on two threads beside 16 and 32 positions, 16-byte
        # packets on four threads beside 8; the batch tile at its smallest
        # block (a 4-image batch is searched from 1).
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 32, 16],
            [1, 32, 32],
            [1, 32, 8],
        ]
        assert [seed["num_threads"] for seed in seeds] == [
            [1, 2, 16],
            [1, 2, 32],
            [1, 4, 8],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        for seed in seeds:
            assert seed["loop_orders"] == [[1, 2, 0]]
            assert seed["cute_lane_layouts"] == ["blocked"] * 3
            # No loads-first unroll: one packet per thread has nothing to
            # unroll, and the knob stays closed where only these seeds fire.
            assert "cute_lane_unroll" not in seed
        generation = spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
        for seed in seeds:
            assert any(seed.items() <= config.items() for config in transferred)
        # Each seed's thread stores its packet with one flush: a 32-byte one
        # and a 16-byte one.
        for seed, lanes in ((seeds[0], 16), (seeds[2], 8)):
            code = bound.to_code(spec.normalized_config(helion.Config(**seed)))
            assert code.count("_cute_store_u16_vec(") == 1
            assert f"cutlass.range_constexpr({lanes})" in code
            assert "_lane_unroll_loads_" not in code


def test_a_store_that_takes_no_packets_gets_packets_of_the_load() -> None:
    """Channels-last to NCHW with 49 positions: the stored rows hold no
    packet, the loaded rows of 96 channels do.  Every thread loads one
    packet of channels and stores it at one position; a warp of position
    threads leads the thread order, so each scalar store of a warp writes 32
    consecutive positions."""
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last, (torch.empty(4, 49, 96, dtype=torch.bfloat16),)
        )
        spec = bound.config_spec
        assert _NAME in spec.autotuner_heuristics
        seeds = _batched_seeds(bound)
        # 32 position threads beside one and two threads of 32-byte packets
        # and two of 16-byte ones.
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 32, 16],
            [1, 32, 32],
            [1, 32, 16],
        ]
        assert [seed["num_threads"] for seed in seeds] == [
            [1, 32, 1],
            [1, 32, 2],
            [1, 32, 2],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 1, 16],
            [1, 1, 16],
            [1, 1, 8],
        ]
        # The store's contiguous tile, the positions, leads here too.
        assert [seed["loop_orders"] for seed in seeds] == [[[1, 2, 0]]] * 3
        generation = spec.create_config_generation()
        transferred = [
            dict(config.config) for _, config in generation.seed_flat_config_pairs()
        ]
        for seed in seeds:
            assert any(seed.items() <= config.items() for config in transferred)
        code = bound.to_code(spec.normalized_config(helion.Config(**seeds[0])))
        assert code.count("cute.arch.load(") == 1
        assert "cutlass.range_constexpr(16)" in code
        assert "_cute_store_u16_vec(" not in code

        # Before SM100 the 16-byte packets alone.
        bound = cpu_bind(
            _from_channels_last,
            (torch.empty(4, 49, 96, dtype=torch.bfloat16),),
            capability=(9, 0),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 1, 8]]

        # A load one element into its storage holds no aligned packet.
        offset = torch.empty(4 * 49 * 96 + 1, dtype=torch.bfloat16)[1:]
        bound = cpu_bind(_from_channels_last, (offset.view(4, 49, 96),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_both_sides_packets_seed_the_stores_first() -> None:
    # 48 positions and 96 channels: either side's rows hold 32-byte packets.
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last, (torch.empty(4, 48, 96, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
        # A masked store keeps the load's seeds.
        masked = cpu_bind(
            _channels_last_masked, (torch.empty(4, 96, 48, dtype=torch.bfloat16),)
        )
        masked_seeds = _batched_seeds(masked)
    assert [seed["cute_vector_widths"] for seed in seeds] == [
        [1, 16, 1],
        [1, 16, 1],
        [1, 8, 1],
        [1, 1, 16],
        [1, 1, 16],
        [1, 1, 8],
    ]
    assert [seed["cute_vector_widths"] for seed in masked_seeds] == [
        [1, 1, 16],
        [1, 1, 16],
        [1, 1, 8],
    ]


def test_a_batched_fp32_seed_holds_eight_channels_per_packet() -> None:
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 64, 50),))
        seeds = _batched_seeds(bound)
        assert [seed["block_sizes"] for seed in seeds] == [
            [1, 16, 16],
            [1, 16, 32],
            [1, 16, 8],
        ]
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 8, 1],
            [1, 8, 1],
            [1, 4, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
    assert code.count("_cute_store_u32_vec(") == 1


def test_four_tiles_keep_both_batch_tiles_at_their_smallest_block() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            _nchw_to_nhwc, (torch.empty(4, 96, 7, 7, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
    # The width tile is the load's contiguous one: 7 positions take 8
    # threads in every seed, so two of the three shapes coincide.
    assert [seed["block_sizes"] for seed in seeds] == [[1, 32, 1, 8], [1, 32, 1, 8]]
    assert [seed["num_threads"] for seed in seeds] == [[1, 2, 1, 8], [1, 4, 1, 8]]
    assert [seed["cute_vector_widths"] for seed in seeds] == [
        [1, 16, 1, 1],
        [1, 8, 1, 1],
    ]
    # Channels, width, then the batch tiles.
    assert [seed["loop_orders"] for seed in seeds] == [[[1, 3, 0, 2]]] * 2


def test_larger_smallest_blocks_take_more_threads() -> None:
    """A large grid starts the search at larger blocks
    (``raise_grid_block_minimums``): the batch tile takes a thread per
    element of its smallest block and the 16-byte seed's position tile its
    smallest block.  (Rows of 4100 positions hold no packets of the load.)"""
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last, (torch.empty(256, 2048, 4100, dtype=torch.bfloat16),)
        )
        seeds = _batched_seeds(bound)
    assert [seed["block_sizes"] for seed in seeds] == [
        [8, 32, 16],
        [8, 32, 32],
        [8, 32, 16],
    ]
    assert [seed["num_threads"] for seed in seeds] == [
        [8, 2, 16],
        [8, 2, 32],
        [8, 4, 16],
    ]


def test_a_smallest_block_past_the_cta_gets_no_seed() -> None:
    """Every thread takes one packet of the channel tile's smallest block
    and one position beside it.  Channel tiles of 512 or more take 32 x 16
    and 64 x 8 threads, a seed's whole CTA, and 32 x 32 would be past it:
    that shape gets no seed and the other two keep theirs.  Channel tiles
    of 1024 or more leave no shape a CTA.  (Rows of 50 positions hold no
    packets of the load.)"""
    x = torch.empty(2, 2048, 50, dtype=torch.bfloat16)
    with cpu_render_target():
        bound = cpu_bind(_channels_last_wide_channel_tiles, (x, 512))
        seeds = _batched_seeds(bound)
        # The registered channel block comes first.
        assert [seed["block_sizes"] for seed in seeds] == [[512, 1, 16], [512, 1, 8]]
        assert [seed["num_threads"] for seed in seeds] == [[32, 1, 16], [64, 1, 8]]
        bound = cpu_bind(_channels_last_wide_channel_tiles, (x, 1024))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_a_16_byte_target_gets_the_16_byte_seed_alone() -> None:
    # Before SM100 a packet is 16 bytes at most.
    with cpu_render_target():
        bound = cpu_bind(
            _channels_last,
            (torch.empty(4, 96, 50, dtype=torch.bfloat16),),
            capability=(9, 0),
        )
        seeds = _batched_seeds(bound)
    assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
    assert [seed["num_threads"] for seed in seeds] == [[1, 4, 8]]


def test_batched_seeds_need_packets_the_lowering_proves() -> None:
    """72-channel bf16 rows (144 bytes) hold 16-byte packets only; 60-channel
    rows (120 bytes) hold none, as do the 289-position rows the two
    BatchNorm kernels of ``pointwise_06377502b8f5`` store into, a base one
    element into its storage, and a view that ends inside a packet.  (Rows
    of 50 positions hold no packets of the load.)"""
    bf16 = torch.bfloat16
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 72, 50, dtype=bf16),))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last, (torch.empty(4, 60, 50, dtype=bf16),))
        assert _NAME not in bound.config_spec.autotuner_heuristics

        # ``out`` as an argument: aligned, it gets the seeds.
        x = torch.empty(4, 96, 50, dtype=bf16)
        bound = cpu_bind(_channels_last_into, (x, torch.empty(4, 50, 96, dtype=bf16)))
        assert len(_batched_seeds(bound)) == 3
        offset = torch.empty(4 * 50 * 96 + 1, dtype=bf16)[1:].view(4, 50, 96)
        bound = cpu_bind(_channels_last_into, (x, offset))
        assert _NAME not in bound.config_spec.autotuner_heuristics
        # 88 of 96 channels: the rows are 32-byte aligned, but the tile's 88
        # channels are whole 16-byte packets only.
        narrow = torch.empty(4, 50, 96, dtype=bf16)[:, :, :88]
        bound = cpu_bind(
            _channels_last_into, (torch.empty(4, 88, 50, dtype=bf16), narrow)
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1


def test_batched_seeds_need_tiles_that_begin_on_a_packet() -> None:
    """A tile's packets start at its begin plus multiples of their width:
    channel tiles that begin at 16 keep every seed, those that begin at 8
    the 16-byte one, and those that begin at 3 none (the lowering stores
    their channels one by one).  The same for the packets of a load.  (Rows
    of 50 positions hold no packets of the other side.)"""
    bf16 = torch.bfloat16
    wide = torch.empty(3, 128, 50, dtype=bf16)
    last = torch.empty(3, 50, 128, dtype=bf16)
    with cpu_render_target():
        bound = cpu_bind(_channels_last_window, (wide, last, 16, 112))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last_window, (wide, last, 8, 104))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 8, 1]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1

        bound = cpu_bind(_channels_last_window, (wide, last, 3, 99))
        assert _NAME not in bound.config_spec.autotuner_heuristics

        bound = cpu_bind(_from_channels_last_window, (last, wide, 8, 104))
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [[1, 1, 8]]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("cute.arch.load(") == 1

        bound = cpu_bind(_from_channels_last_window, (last, wide, 3, 99))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_a_masked_load_gets_no_packets_of_its_own() -> None:
    """A masked load keeps a packet only where its mask is uniform over the
    packet: the first 40 of 96 channels end inside a 32-byte packet, whose
    channels the lowering then loads one by one.  A seed does not prove a
    mask, so a masked load gets no load-packet seed (rows of 50 positions
    hold no packets of the store).  The packets of the store's tile do not
    depend on it: rows of 48 positions keep the three store seeds."""
    bf16 = torch.bfloat16
    with cpu_render_target():
        bound = cpu_bind(
            _from_channels_last_masked_load, (torch.empty(4, 50, 96, dtype=bf16), 40)
        )
        assert _NAME not in bound.config_spec.autotuner_heuristics

        bound = cpu_bind(
            _from_channels_last_masked_load, (torch.empty(4, 48, 96, dtype=bf16), 40)
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1


def test_a_shifted_subscript_gets_no_packets_of_its_own() -> None:
    """A subscript that adds a scalar to a tile's index reads as the tile in
    the access's fact, and the lowering of a tile's lane packets admits it
    for loads only.  A store with one keeps its lanes scalar, on its
    contiguous dim (``tile_c.index + 16``) and on another
    (``tile_s.index + 2``): it gets no store-packet seed, and the load keeps
    its three.  A load with one on its contiguous dim keeps the packets the
    scalar leaves aligned, which a seed does not prove: it gets no
    load-packet seed, and the store keeps its three.  One on another dim of
    the load moves no packet: all six."""
    bf16 = torch.bfloat16
    store_seeds = [[1, 16, 1], [1, 16, 1], [1, 8, 1]]
    load_seeds = [[1, 1, 16], [1, 1, 16], [1, 1, 8]]
    with cpu_render_target():
        x = torch.empty(4, 96, 48, dtype=bf16)
        for lane, shape in ((True, (4, 48, 112)), (False, (4, 50, 96))):
            bound = cpu_bind(
                _channels_last_shifted_store, (x, torch.empty(shape, dtype=bf16), lane)
            )
            seeds = _batched_seeds(bound)
            assert [seed["cute_vector_widths"] for seed in seeds] == load_seeds
            code = bound.to_code(
                bound.config_spec.normalized_config(helion.Config(**seeds[0]))
            )
            assert code.count("cute.arch.load(") == 1
            assert "_cute_store_u16_vec(" not in code

        out = torch.empty(4, 96, 48, dtype=bf16)
        bound = cpu_bind(
            _from_channels_last_shifted_load,
            (torch.empty(4, 48, 112, dtype=bf16), out, True),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == store_seeds
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[0]))
        )
        assert code.count("_cute_store_u16_vec(") == 1
        assert "cute.arch.load(" not in code

        bound = cpu_bind(
            _from_channels_last_shifted_load,
            (torch.empty(4, 50, 96, dtype=bf16), out, False),
        )
        seeds = _batched_seeds(bound)
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            *store_seeds,
            *load_seeds,
        ]
        code = bound.to_code(
            bound.config_spec.normalized_config(helion.Config(**seeds[3]))
        )
        assert code.count("cute.arch.load(") == 1


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.float64])
def test_no_batched_seeds_for_dtypes_stored_as_scalars(dtype: torch.dtype) -> None:
    # Tile stores are packets for bf16, fp16 and fp32 only, and the seeds
    # ask the same of the loads: rows of 48 positions and 96 channels would
    # hold the packets of either side.
    with cpu_render_target():
        bound = cpu_bind(_channels_last, (torch.empty(4, 96, 48, dtype=dtype),))
        assert _NAME not in bound.config_spec.autotuner_heuristics


def test_no_batched_seeds_without_a_layout_change_the_seeds_serve() -> None:
    bf16 = torch.bfloat16
    with cpu_render_target():
        for kernel, args in (
            # The same tile contiguous on both sides.
            (_batched_scale, (torch.empty(4, 96, 48, dtype=bf16),)),
            (
                _channels_last,
                (torch.empty(4, 48, 96, dtype=bf16).permute(0, 2, 1),),
            ),
            # A masked store takes no packets (and rows of 50 positions hold
            # none of the load).
            (_channels_last_masked, (torch.empty(4, 96, 50, dtype=bf16),)),
            # No loop order spans the three tiles (the channel tiles, whose
            # packets the store would take, are the root loop's).
            (
                _channels_last_per_channel_tile,
                (torch.empty(4, 96, 48, dtype=bf16),),
            ),
            # The three tiles of a loop under a root loop: its threads and
            # packets are not a launch grid's.
            (
                _channels_last_under_a_loop,
                (torch.empty(4, 96, 48, dtype=bf16), torch.empty(4)),
            ),
            # Dynamic shapes: no static extent to prove the packets on.
            (_channels_last_dynamic, (torch.empty(4, 96, 48, dtype=bf16),)),
        ):
            bound = cpu_bind(kernel, args)
            assert _NAME not in bound.config_spec.autotuner_heuristics


def test_batched_seeds_need_a_kernel_that_renders_their_layout() -> None:
    """The admission the batched seeds share with the load-packet seed: a
    pointwise kernel whose only blocks with threads are the store's tiles
    and which stores into no tensor it loads.
    Every refusal is ``_channels_last`` with one thing beside it; alone it
    gets the six seeds, at either shape."""
    bf16 = torch.bfloat16
    x = torch.empty(4, 96, 48, dtype=bf16)
    square = torch.empty(4, 96, 96, dtype=bf16)
    with cpu_render_target():
        for alone in (x, square):
            assert len(_batched_seeds(cpu_bind(_channels_last, (alone,)))) == 6
        bound = cpu_bind(_channels_last_of_a_sum, (square, square.clone(), False))
        assert len(_batched_seeds(bound)) == 6
        for kernel, args in (
            # Not pointwise: beside a reduction of the tile's lanes four of
            # the six seeds do not render.  An atomic add of the whole tile
            # keeps the layouts and is declined with it on purpose.
            (_channels_last_and_channel_sums, (x,)),
            (_channels_last_and_an_atomic, (x,)),
            # The untiled dim of whole rows, copied or summed, takes thread
            # axis 0: CTAs of up to 1024 threads.
            (_channels_last_beside_rows, (x, torch.empty(4, 8), False)),
            (_channels_last_beside_rows, (x, torch.empty(4, 64), True)),
            # Another root loop's threads: the store-packet seeds render no
            # packet store.
            (_channels_last_beside_a_loop, (x, torch.empty(1000))),
            # A store into a tensor the kernel loads: channels-last into
            # itself, which the code generator refuses, for the first load
            # and for a second one; and as it was loaded, which renders the
            # layout and is declined on purpose (no in-place kernel was
            # measured).
            (_channels_last_in_place, (square, True)),
            (_channels_last_of_a_sum, (square, square.clone(), True)),
            (_channels_last_in_place, (x, False)),
        ):
            bound = cpu_bind(kernel, args)
            assert _NAME not in bound.config_spec.autotuner_heuristics


def test_batched_seeds_leave_the_search_space_as_it_was() -> None:
    """The seeds open no knob: a shape they decline (60-channel rows of 50
    positions) has the search space of one they serve."""
    with cpu_render_target():
        served = cpu_bind(
            _channels_last, (torch.empty(4, 96, 48, dtype=torch.bfloat16),)
        )
        declined = cpu_bind(
            _channels_last, (torch.empty(4, 60, 50, dtype=torch.bfloat16),)
        )
        assert len(_batched_seeds(served)) == 6
        assert _NAME not in declined.config_spec.autotuner_heuristics
        assert (
            served.config_spec.structural_fingerprint()
            == declined.config_spec.structural_fingerprint()
        )


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteTransposedStoreSeeds(TestCase):
    # One seed per dtype: the seeds of a dtype differ in tile shape only.
    def test_seeds_are_bitwise_torch(self) -> None:
        # 1021 columns padded to 1100 and 200 rows: partial tiles along both
        # dims, with either store first.
        x = torch.randn(200, 1021, device=DEVICE).to(torch.bfloat16)
        expected = torch.nn.functional.pad(x, (0, 79))
        for transpose_first in (True, False):
            args = (x, 1100, transpose_first)
            seed = _seeds(_pad_and_transpose.bind(args), 16)[0]
            _, (out_t, out) = code_and_output(_pad_and_transpose, args, **seed)
            torch.testing.assert_close(out_t, expected.t(), rtol=0, atol=0)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_fp32_and_int8_seeds_are_bitwise_torch(self) -> None:
        # Partial tiles along both dims: 100 (and 96) rows, 520 columns.
        for x, packet in (
            (torch.randn(100, 520, device=DEVICE), 8),
            (torch.randint(-32, 32, (96, 520), device=DEVICE, dtype=torch.int8), 16),
        ):
            seed = _seeds(_scaled_transpose.bind((x,)), packet)[0]
            _, out = code_and_output(_scaled_transpose, (x,), **seed)
            torch.testing.assert_close(out, (x * 2).t(), rtol=0, atol=0)

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_the_load_packet_seed_is_bitwise_torch(self) -> None:
        # Partial tiles along both dims: 200 rows in tiles of 32, 528 columns
        # in tiles of 64 (fp32: the last tile holds two of its eight packets)
        # and of 128 (bf16: one of its eight), into rows padded to 208.
        bf16 = torch.bfloat16
        seeds = {}
        for dtype in (torch.float32, bf16):
            x = torch.randn(200, 528, device=DEVICE).to(dtype)
            padded = torch.zeros(528, 208, device=DEVICE, dtype=bf16)
            args = (x, padded[:, :200])
            (seeds[dtype],) = _load_packet_seeds(_cast_both_ways.bind(args))
            code, out = code_and_output(_cast_both_ways, args, **seeds[dtype])
            assert "cute.arch.load(" in code
            torch.testing.assert_close(out, x.to(bf16), rtol=0, atol=0)
            torch.testing.assert_close(padded[:, :200], x.to(bf16).t(), rtol=0, atol=0)
            # The padding keeps its zeros.
            assert not padded[:, 200:].any()
        # The fp32 seed pinned on a load whose packets are not proven (one
        # element into its storage): scalar loads.
        x = torch.randn(200 * 528 + 1, device=DEVICE)[1:].view(200, 528)
        padded = torch.zeros(528, 208, device=DEVICE, dtype=bf16)
        args = (x, padded[:, :200])
        assert _load_packet_seeds(_cast_both_ways.bind(args)) == []
        code, out = code_and_output(_cast_both_ways, args, **seeds[torch.float32])
        assert "cute.arch.load(" not in code
        torch.testing.assert_close(out, x.to(bf16), rtol=0, atol=0)
        torch.testing.assert_close(padded[:, :200], x.to(bf16).t(), rtol=0, atol=0)

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_batched_seeds_are_bitwise_torch(self) -> None:
        # Partial tiles along both lane dims: 80 bf16 channels in tiles of
        # 32 (the last tile holds one of its two packets), 72 fp32 channels
        # in tiles of 16, 50 positions in tiles of 16 and 8.  The 32-byte
        # and the 16-byte seed for bf16, the 32-byte one for fp32.
        for x, picks in (
            (torch.randn(3, 80, 50, device=DEVICE).to(torch.bfloat16), (0, 2)),
            (torch.randn(3, 72, 50, device=DEVICE), (0,)),
        ):
            seeds = _batched_seeds(_channels_last.bind((x,)))
            assert len(seeds) == 3
            for pick in picks:
                code, out = code_and_output(_channels_last, (x,), **seeds[pick])
                assert "_vec(" in code
                torch.testing.assert_close(out, x.permute(0, 2, 1), rtol=0, atol=0)
        # The load's packets where the store takes none: 50 positions in
        # tiles of 32 threads, 80 channels in packets of 16 (one thread, and
        # two with the last tile holding one of its two packets).
        x = torch.randn(3, 50, 80, device=DEVICE).to(torch.bfloat16)
        seeds = _batched_seeds(_from_channels_last.bind((x,)))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 1, 16],
            [1, 1, 16],
            [1, 1, 8],
        ]
        for seed in seeds[:2]:
            code, out = code_and_output(_from_channels_last, (x,), **seed)
            assert "cute.arch.load(" in code
            torch.testing.assert_close(out, x.permute(0, 2, 1), rtol=0, atol=0)

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_store_seeds_beside_a_masked_load_are_bitwise_torch(self) -> None:
        # The first 40 of 80 channels, zeros for the others, into rows of 48
        # positions in tiles of 32 and 16 (the last tile of 32 holds one of
        # its two packets): the 32-byte and the 16-byte store seed.
        x = torch.randn(3, 48, 80, device=DEVICE).to(torch.bfloat16)
        expected = x.permute(0, 2, 1).clone()
        expected[:, 40:] = 0
        args = (x, 40)
        seeds = _batched_seeds(_from_channels_last_masked_load.bind(args))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        for seed in (seeds[1], seeds[2]):
            code, out = code_and_output(_from_channels_last_masked_load, args, **seed)
            assert "_cute_store_u16_vec(" in code
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_seeds_beside_a_shifted_subscript_are_bitwise_torch(self) -> None:
        # The packets of the other side beside a subscript that adds a
        # scalar to a tile's index: the load's under a store 16 channels
        # into its rows and under one two rows down (48 positions in tiles
        # of 32: the last tile holds one of its two packets), and the
        # store's under a load of the channels from 16 on.
        bf16 = torch.bfloat16
        x = torch.randn(3, 80, 48, device=DEVICE).to(bf16)
        for lane, shape in ((True, (3, 48, 96)), (False, (3, 50, 80))):
            expected = torch.zeros(shape, device=DEVICE, dtype=bf16)
            if lane:
                expected[:, :, 16:] = x.permute(0, 2, 1)
            else:
                expected[:, 2:] = x.permute(0, 2, 1)
            args = (x, torch.zeros(shape, device=DEVICE, dtype=bf16), lane)
            seeds = _batched_seeds(_channels_last_shifted_store.bind(args))
            assert [seed["cute_vector_widths"] for seed in seeds] == [
                [1, 1, 16],
                [1, 1, 16],
                [1, 1, 8],
            ]
            code, out = code_and_output(_channels_last_shifted_store, args, **seeds[1])
            assert "cute.arch.load(" in code
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        x = torch.randn(3, 48, 96, device=DEVICE).to(bf16)
        args = (x, torch.zeros(3, 80, 48, device=DEVICE, dtype=bf16), True)
        seeds = _batched_seeds(_from_channels_last_shifted_load.bind(args))
        assert [seed["cute_vector_widths"] for seed in seeds] == [
            [1, 16, 1],
            [1, 16, 1],
            [1, 8, 1],
        ]
        code, out = code_and_output(_from_channels_last_shifted_load, args, **seeds[1])
        assert "_cute_store_u16_vec(" in code
        expected = x[:, :, 16:].permute(0, 2, 1)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_a_batched_seed_is_bitwise_torch_on_other_layouts(self) -> None:
        """A seed's config on layouts its packets are not proven on (a
        pinned config meets any input): what the lowering cannot prove it
        stores as scalars."""
        bf16 = torch.bfloat16
        aligned = (
            torch.empty(3, 96, 50, device=DEVICE, dtype=bf16),
            torch.empty(3, 50, 96, device=DEVICE, dtype=bf16),
        )
        wide, narrow = (
            seed
            for seed in _batched_seeds(_channels_last_into.bind(aligned))
            if seed["num_threads"] in ([1, 2, 16], [1, 4, 8])
        )
        values = torch.randn(3, 96, 100, device=DEVICE).to(bf16)
        # The load contiguous along the channels as well (a permuted view),
        # the store one element into its storage: no packet is aligned.
        permuted = values[:, :, :50].permute(0, 2, 1).contiguous().permute(0, 2, 1)
        offset = torch.zeros(3 * 50 * 96 + 1, device=DEVICE, dtype=bf16)[1:]
        # A strided load, and a store into 88 of the 96 channels of a wider
        # buffer: the rows end inside a 32-byte packet (scalar stores) and on
        # a 16-byte one (packets; the last tile holds three of its four).
        strided = values[:, :88, ::2]
        buffer = torch.zeros(3, 50, 96, device=DEVICE, dtype=bf16)
        # The load and the store in one storage.
        shared = torch.zeros(2, 3 * 96 * 50, device=DEVICE, dtype=bf16)
        shared[0] = values[:, :, 50:].reshape(-1)
        for x, out, seed, packets in (
            (permuted, offset.view(3, 50, 96), wide, False),
            (strided, buffer[:, :, :88], wide, False),
            (strided, buffer[:, :, :88], narrow, True),
            (shared[0].view(3, 96, 50), shared[1].view(3, 50, 96), wide, True),
        ):
            expected = x.permute(0, 2, 1).clone()
            code, result = code_and_output(_channels_last_into, (x, out), **seed)
            assert ("_cute_store_u16_vec(" in code) == packets
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        # The channels past the view keep their zeros.
        assert not buffer[:, :, 88:].any()

    @skipIfCudaCapabilityLessThan((10, 0), reason="32-byte packets need SM100")
    def test_batched_seeds_off_a_tile_begin_are_bitwise_torch(self) -> None:
        """Channel tiles that begin at 8 keep the 16-byte seed (88 channels
        in tiles of 32: the last tile holds three of its four packets), for
        the store's packets and for the load's.  Tiles that begin at 3 get
        no seed; a seed of tiles that begin at 0 pinned on them stores
        scalars."""
        bf16 = torch.bfloat16
        values = torch.randn(3, 128, 50, device=DEVICE).to(bf16)
        expected = torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16)
        expected[:, :, 8:96] = values.permute(0, 2, 1)[:, :, 8:96]

        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 8, 96)
        (seed,) = _batched_seeds(_channels_last_window.bind(args))
        assert seed["cute_vector_widths"] == [1, 8, 1]
        code, out = code_and_output(_channels_last_window, args, **seed)
        assert "_cute_store_u16_vec(" in code
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

        last = values.permute(0, 2, 1).contiguous()
        args = (last, torch.zeros(3, 128, 50, device=DEVICE, dtype=bf16), 8, 96)
        (seed,) = _batched_seeds(_from_channels_last_window.bind(args))
        assert seed["cute_vector_widths"] == [1, 1, 8]
        code, out = code_and_output(_from_channels_last_window, args, **seed)
        assert "cute.arch.load(" in code
        torch.testing.assert_close(out, expected.permute(0, 2, 1), rtol=0, atol=0)

        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 0, 96)
        wide = _batched_seeds(_channels_last_window.bind(args))[0]
        assert wide["cute_vector_widths"] == [1, 16, 1]
        args = (values, torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16), 3, 99)
        assert _batched_seeds(_channels_last_window.bind(args)) == []
        code, out = code_and_output(_channels_last_window, args, **wide)
        assert "_cute_store_u16_vec(" not in code
        expected = torch.zeros(3, 50, 128, device=DEVICE, dtype=bf16)
        expected[:, :, 3:99] = values.permute(0, 2, 1)[:, :, 3:99]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
