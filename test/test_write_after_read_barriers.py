"""Barriers before a write that may overwrite what another thread has yet to
read, or has stored.

A load issued by one thread and a later store, atomic or snippet of another
thread to the same element are a write-after-read race on global memory: the
write may land first.  Two stores of different threads to one element are a
write-after-write race: the earlier may land last.  On Triton the later write
follows a CTA barrier (``mark_intra_loop_raw_barriers`` with ``reads``), and
so does a sibling loop that writes what an earlier one read or stored
(``needs_barrier_before``).  An update of what a load of the same body read,
at its index and computed from it (``x[t] = f(x[t])``), touches only the
elements each thread read itself and takes none, as do stores of disjoint
elements.  The kernels render on CPU (TileIR emits no barriers, and ref-eager
mode generates no code); the last tests run two across warps.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compat import gluon_api_available
from helion._compiler import device_function
from helion._compiler.indexing_strategy import TensorDescriptorIndexingStrategy
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfFn
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessTritonBarriers
from helion._testing.cute_codegen import cpu_bind
from helion._testing.triton_codegen import gluon_cpu_codegen
import helion.language as hl
from helion.language import loops
from helion.runtime.settings import _get_backend

pytestmark = pytest.mark.cpu_only

_TRITON_BARRIER = "tl.debug_barrier()"


def _overwrites_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        # Each thread reads the row's upper half, which other threads own in
        # the whole-row store.
        upper = x[tile_m, half + 64]
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = upper
    return out


def _updates_its_own_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x[tile_m, :] * 2.0 + 1.0
    return x


def _resets_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = row
    return out


def _gathers_and_updates(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(indices.size(0)):
        # Repeated indices put one element in several threads.
        idx = indices[tile]
        x[idx] = x[idx] + 1.0
    return x


def _updates_a_repeating_index(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        # Positions k and k + 32 of the index name one element.
        idx = hl.arange(64) % 32
        x[tile_m, idx] = x[tile_m, idx] + 1.0
    return x


def _doubles_its_arange_columns(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        idx = hl.arange(128)
        x[tile_m, idx] = x[tile_m, idx] * 2.0
    return x


def _doubles_a_broadcast_column(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        # ``x`` has one column, over which the index broadcasts: every
        # position names that one element.
        idx = hl.arange(128)
        x[tile_m, idx] = x[tile_m, idx] * 2.0
    return x


def _doubles_a_broadcast_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile(y.size()):
        # ``x`` has one column, over which the tile broadcasts.
        x[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return x


def _updates_a_wrapping_index(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        # Positions k and k + 256 wrap to one column.
        idx = hl.arange(512, dtype=torch.uint8)
        x[tile_m, idx] = x[tile_m, idx] + 1.0
    return x


def _adds_its_own_columns(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        idx = hl.arange(128)
        hl.atomic_add(x, [tile_m, idx], x[tile_m, idx])
    return x


def _resets_what_it_added_into(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        # The reset must not land before the add, whose result goes unused.
        hl.atomic_add(x, [tile_m, hl.arange(64) + 64], x.new_ones([tile_m, 64]))
        x[tile_m, :] = x.new_zeros([tile_m, 128])
    return x


def _updates_an_offset_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(64):
            idx = tile_n.index + 64
            x[tile_m, idx] = x[tile_m, idx] * 2.0
    return x


def _updates_a_scaled_arange(x: torch.Tensor, alpha: int) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        # ``alpha`` scales the arange; with 0 every position is one column.
        idx = torch.add(0, hl.arange(128), alpha=alpha)
        x[tile_m, idx] = x[tile_m, idx] + 1.0
    return x


def _transposes_its_own_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_a, tile_b in hl.tile([x.size(0), x.size(1)], block_size=[32, 32]):
        # The thread that stores an element may have loaded its mirror.
        x[tile_a, tile_b] = x[tile_a, tile_b].T
    return x


def _updates_a_strided_arange(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        idx = hl.arange(0, 128, 2)
        x[tile_m, idx] = x[tile_m, idx] * 2.0
    return x


def _scales_its_rows_in_a_loop(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        row_scale = scale[tile_m]
        for tile_n in hl.tile(n):
            x[tile_m, tile_n] = x[tile_m, tile_n] * row_scale[:, None]
    return x


def _updates_and_sums_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        x[tile_m, :] = row * 2.0
        # The sum may combine another thread's copy of the row.
        out[tile_m] = row.sum(-1)
    return out


def _resets_what_an_atomic_returned(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        old = hl.atomic_add(x, [tile_m, half + 64], x.new_ones([tile_m, 64]))
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = old
    return out


def _overwrites_what_a_packed_load_read(
    x: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([offsets.size(0), 16], dtype=torch.float16, device=x.device)
    for tile in hl.tile(offsets.size(0)):
        # A packed load reads 16 values anywhere in x, which other threads
        # own in the whole-tensor store.
        lanes = hl.load_bfloat16_x16_to_float16(x, offsets[tile])
        x[:] = x.new_zeros([128])
        for i in hl.static_range(16):
            out[tile, i] = lanes[i]
    return out


def _packed_load_after_a_store(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    out = torch.empty([offsets.size(0), 16], dtype=torch.float16, device=x.device)
    for tile in hl.tile(offsets.size(0)):
        x[:] = x.new_zeros([128])
        # It reads values other threads stored.
        lanes = hl.load_bfloat16_x16_to_float16(x, offsets[tile])
        for i in hl.static_range(16):
            out[tile, i] = lanes[i]
    return out


def _overwrites_in_a_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            mirrored = x[tile_m, n - 1 - tile_n.index]
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
            out[tile_m, tile_n] = mirrored
    return out


def _updates_in_a_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            x[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return x


def _sibling_loop_overwrites(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n, block_size=32):
            out[tile_m, tile_n] = x[tile_m, n - 1 - tile_n.index]
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
    return out


def _sibling_loop_overwrites_an_atomic_target(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n, block_size=32):
            hl.atomic_add(x, [tile_m, tile_n], x.new_ones([tile_m, tile_n]))
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
    return x


def _reads_before_the_loop_it_overwrites_in(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, 64], device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        upper = x[tile_m, hl.arange(64) + 64]
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
        out[tile_m, :] = upper
    return out


def _snippet_overwrites_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        upper = x[tile_m, hl.arange(64) + 64]
        hl.inline_triton(
            "tl.store({0} + {1}[:, None] * 128 + tl.arange(0, 128)[None, :], 0.0)",
            args=(x, tile_m.index),
            output_like=None,
        )
        out[tile_m, :] = upper
    return out


def _overwrites_what_a_snippet_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        upper = hl.inline_triton(
            "tl.load({0} + {1}[:, None] * 128 + 64 + tl.arange(0, 64)[None, :])",
            args=(x, tile_m.index),
            output_like=out[tile_m, :],
        )
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = upper
    return out


def _resets_an_atomic_target_after_a_bare_barrier(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile_m, hl.arange(64) + 64], x.new_ones([tile_m, 64]))
        upper = y[tile_m, hl.arange(64) + 64]
        y[tile_m, :] = y.new_zeros([tile_m, 128])
        # The bare barrier before the store to ``y`` leaves the add pending.
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = upper
    return out


def _rereads_what_it_overwrote(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        upper = x[tile_m, hl.arange(64) + 64]
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = upper + x[tile_m, hl.arange(64)]
    return out


def _updates_its_own_block(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile(x.size()):
        x[tile_m, tile_n] = x[tile_m, tile_n] * 2.0 + 1.0
    return x


def _updates_its_own_blocks_in_a_loop(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            x[tile_m, tile_n] = x[tile_m, tile_n] * 2.0 + 1.0
    return x


def _overwrites_a_store_of_another_graph(
    x: torch.Tensor, out: torch.Tensor, flag: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        if flag[0] > 0:
            for tile_n in hl.tile(x.size(1)):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
        out[tile_m, :] = x[tile_m, :] * 3.0
    return out


def _reads_what_it_overwrote(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = x[tile_m, hl.arange(64) + 64]
    return out


def _adds_into_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        upper = x[tile_m, half + 64]
        hl.atomic_add(x, [tile_m, half + 64], x.new_ones([tile_m, 64]))
        out[tile_m, :] = upper
    return out


def _releases_into_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        upper = x[tile_m, half + 64]
        hl.atomic_add(x, [tile_m, half + 64], x.new_ones([tile_m, 64]), sem="release")
        out[tile_m, :] = upper
    return out


def _restores_part_of_what_it_stored(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_ones([tile_m, 128])
        # Other threads own the upper half in the whole-row store.
        x[tile_m, hl.arange(64) + 64] = x.new_full([tile_m, 64], 2.0)
    return x


def _stores_twice(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_ones([tile_m, 128])
        x[tile_m, :] = x.new_full([tile_m, 128], 2.0)
    return x


def _stores_the_halves_of_a_row(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        x[tile_m, half] = x.new_ones([tile_m, 64])
        x[tile_m, half + 64] = x.new_zeros([tile_m, 64])
    return x


def _stores_sliced_halves(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, 0:64] = x.new_ones([tile_m, 64])
        x[tile_m, 64:128] = x.new_zeros([tile_m, 64])
    return x


def _doubles_one_half_into_the_other(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        lo = x[tile_m, 0:64]
        x[tile_m, 64:128] = lo * 2.0
    return x


def _doubles_one_arange_half_into_the_other(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        x[tile_m, half + 64] = x[tile_m, half] * 2.0
    return x


def _stores_two_slots(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, 0, :] = x.new_ones([tile_m, 128])
        x[tile_m, 1, :] = x.new_zeros([tile_m, 128])
    return x


def _stores_a_column_at_its_negative_index(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, hl.arange(128)] = x.new_ones([tile_m, 128])
        # Column -1 is column 127.
        x[tile_m, -1] = x.new_zeros([tile_m])
    return x


def _stores_columns_a_wrap_apart(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        columns = hl.arange(64)
        x[tile_m, columns] = x.new_ones([tile_m, 64])
        # Two 32-bit wraps: the same columns.
        x[tile_m, columns + 2147483647 + 2147483647 + 2] = x.new_zeros([tile_m, 64])
    return x


def _stores_wrapping_halves(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, hl.arange(128, dtype=torch.uint8)] = x.new_ones([tile_m, 128])
        # 224 + 32 wraps to 0: the upper half's tail lands on the first.
        upper = hl.arange(64, dtype=torch.uint8) + 224
        x[tile_m, upper] = x.new_zeros([tile_m, 64])
    return x


def _stores_the_halves_of_a_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), 64]):
        x[tile_m, tile_n] = x.new_ones([tile_m, tile_n])
        # The mask stops a block wider than 64 short of the upper half.
        x[tile_m, tile_n.index + 64] = x.new_zeros([tile_m, tile_n])
    return x


def _stores_broadcast_halves_of_a_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), 64]):
        rows, columns = tile_m.index[:, None], tile_n.index
        x[rows, columns[None, :]] = x.new_ones([tile_m, tile_n])
        x[rows, (columns + 64)[None, :]] = x.new_zeros([tile_m, tile_n])
    return x


def _stores_rows_a_half_apart(x: torch.Tensor) -> torch.Tensor:
    half = x.size(0) // 2
    for tile_m in hl.tile(half, block_size=1):
        x[tile_m.begin, :] = x.new_ones([128])
        x[tile_m.begin + half, :] = x.new_zeros([128])
    return x


def _stores_the_halves_of_gathered_rows(
    x: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    for tile_t in hl.tile(rows.size(0)):
        row = rows[tile_t]
        half = hl.arange(64)
        x[row[:, None], half[None, :]] = x.new_ones([tile_t, 64])
        x[row[:, None], (half + 64)[None, :]] = x.new_zeros([tile_t, 64])
    return x


def _sibling_loop_restores(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_ones([tile_m, tile_n])
        for tile_n in hl.tile(n, block_size=64):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
    return x


def _copies_twice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile(x.size()):
        value = y[tile_m, tile_n]
        x[tile_m, tile_n] = value
        x[tile_m, tile_n] = value * 2.0
    return x


def _stores_its_rows_each_iteration(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_k in hl.tile(y.size(1)):
            # Each iteration overwrites the block the one before stored, from
            # the same threads.
            x[tile_m, 0:64] = y[tile_m, tile_k]
    return x


def _stores_a_row_then_its_tiles(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_ones([tile_m, x.size(1)])
        for tile_n in hl.tile(x.size(1)):
            # Other threads stored the row's elements before the loop.
            x[tile_m, tile_n] = y[tile_m, tile_n]
    return x


def _stores_its_tiles_in_two_loops(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            x[tile_m, tile_n] = y[tile_m, tile_n] * 2.0
        for tile_n in hl.tile(x.size(1)):
            # Other threads stored these elements in the first loop.
            x[tile_m, tile_n] = y[tile_m, tile_n] + 1.0
    return x


def _restores_rows_beside_another_loop(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        for tile_k in hl.tile(y.size(1)):
            # The first store overwrites the second's of the iteration before,
            # which is not lowered yet when its barrier is.
            x[tile_m, 0:64] = y[tile_m, tile_k]
            x[tile_m, 32:96] = y[tile_m, tile_k] * 2.0
        for tile_n in hl.tile(y.size(1)):
            out[tile_m, tile_n] = y[tile_m, tile_n] + 1.0
    return out


def _stores_and_reloads_a_row_beside_a_loop(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    z = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(y.size(1)):
            out[tile_m, tile_n] = y[tile_m, tile_n] + 1.0
        x[tile_m, :] = y[tile_m, :] * 2.0
        # Other threads stored the row: a full barrier.
        z[tile_m, :] = x[tile_m, :] + 1.0
    return out


def _stores_then_rereads_its_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
        for tile_n in hl.tile(x.size(1)):
            # The first 16 columns, which the first iteration stores.
            first = y[tile_m, 0:16].sum(-1, keepdim=True)
            y[tile_m, tile_n] = first + x[tile_m, tile_n]
    return out


def _stores_overlapping_rows_each_iteration(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_k in hl.tile(y.size(1)):
            # The first store overwrites the second's of the iteration before.
            x[tile_m, 0:64] = y[tile_m, tile_k]
            x[tile_m, 32:96] = y[tile_m, tile_k] * 2.0
    return x


def _stores_reversed_every_other_iteration(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        columns = hl.arange(64)
        for tile_k in hl.tile(y.size(1)):
            # The loop carries the index: other threads store the columns.
            x[tile_m, columns] = y[tile_m, tile_k]
            columns = 63 - columns
    return x


def _stores_repeating_columns_each_iteration(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        columns = hl.arange(128) % 64
        for tile_k in hl.tile(y.size(1)):
            # Two threads store each column: which one last is unordered.
            x[tile_m, columns] = y[tile_m, tile_k]
    return x


def _normalizes_its_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        scale = torch.rsqrt((row * row).mean(-1, keepdim=True) + 1e-6)
        # Rolled, each iteration stores its own chunk of the row.
        out[tile_m, :] = row * scale + y[tile_m, :]
    return out


# The arguments of the kernels that take other than one 32 x 128 tensor.
_ARGS: dict[object, tuple[object, ...]] = {
    _gathers_and_updates: (torch.randn(64), torch.randint(0, 64, [32])),
    _scales_its_rows_in_a_loop: (torch.randn(32, 128), torch.randn(32)),
    _updates_a_wrapping_index: (torch.randn(32, 256),),
    _updates_a_scaled_arange: (torch.randn(32, 128), 0),
    _doubles_a_broadcast_column: (torch.randn(32, 1),),
    _doubles_a_broadcast_tile: (torch.randn(32, 1), torch.randn(32, 128)),
    _resets_an_atomic_target_after_a_bare_barrier: (
        torch.randn(32, 128),
        torch.randn(32, 128),
    ),
    _transposes_its_own_tile: (torch.randn(64, 64),),
    _overwrites_what_a_packed_load_read: (
        torch.randn(128).bfloat16(),
        torch.randint(0, 8, [8]),
    ),
    _packed_load_after_a_store: (torch.randn(128).bfloat16(), torch.randint(0, 8, [8])),
    _stores_two_slots: (torch.randn(32, 2, 128),),
    _stores_the_halves_of_gathered_rows: (
        torch.randn(64, 128),
        torch.randint(0, 64, [16]),
    ),
}


# The bound kernels, by kernel, backend, the TMA and warp-specialization
# support checks the tests patch (each patch a new stand-in), and argument
# layouts (``_layouts``): each binds once per module.
_BOUND: dict[tuple[object, ...], object] = {}


def _layouts(args: tuple[object, ...]) -> tuple[object, ...]:
    """What binding specializes on: each tensor's sizes, strides, offset,
    dtype and the first argument sharing its storage; other values."""
    storages = [
        arg.untyped_storage().data_ptr() if isinstance(arg, torch.Tensor) else None
        for arg in args
    ]
    return tuple(
        (
            tuple(arg.shape),
            arg.stride(),
            arg.storage_offset(),
            arg.dtype,
            storages.index(storages[i]),
        )
        if isinstance(arg, torch.Tensor)
        else arg
        for i, arg in enumerate(args)
    )


def _bind(
    fn: object, backend: str | None = None, args: tuple[object, ...] = ()
) -> object:
    if not args:
        args = _ARGS.get(fn, (torch.randn(32, 128),))
    key = (
        fn,
        backend,
        loops._supports_warp_specialize,
        device_function._selects_tma_access,
        TensorDescriptorIndexingStrategy.is_supported,
        *_layouts(args),
    )
    if key not in _BOUND:
        options = {} if backend is None else {"backend": backend}
        _BOUND[key] = cpu_bind(helion.kernel(fn, static_shapes=True, **options), args)
    return _BOUND[key]


def _code(fn: object, args: tuple[object, ...] = ()) -> str:
    with cpu_render_target():
        bound = _bind(fn, args=args)
        default = bound.config_spec.default_config()
        # A row per program, a whole row per access.
        return bound.to_code(
            helion.Config(
                **{**default.config, "block_sizes": [1] * len(default.block_sizes)}
            )
        )


def _kernel_body(code: str) -> list[ast.stmt]:
    kernel = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    return kernel.body


def _is_barrier(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and ast.unparse(statement) == _TRITON_BARRIER


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "read", "write"),
    [
        # A store, an atomic and a snippet over other threads' reads, and a
        # store over a snippet's.
        (_overwrites_what_it_read, "tl.load(x", "tl.store(x"),
        (_adds_into_what_it_read, "tl.load(x", "tl.atomic_add(x"),
        (_snippet_overwrites_what_it_read, "tl.load(x", "tl.store(x"),
        (_overwrites_what_a_snippet_read, "tl.load(x", "tl.store(x"),
        # The reset's value is not computed from the read.
        (_resets_what_it_read, "tl.load(x", "tl.store(x"),
        # Indices that may name one element twice: loaded, repeating,
        # broadcast over a size-1 dimension, wrapping, scaled by alpha, or
        # strided (a step may wrap; only unit steps count as distinct).
        (_gathers_and_updates, "tl.load(x", "tl.store(x"),
        (_doubles_a_broadcast_column, "tl.load(x", "tl.store(x"),
        (_doubles_a_broadcast_tile, "tl.load(x", "tl.store(x"),
        (_updates_a_repeating_index, "tl.load(x", "tl.store(x"),
        (_updates_a_wrapping_index, "tl.load(x", "tl.store(x"),
        (_updates_a_scaled_arange, "tl.load(x", "tl.store(x"),
        (_updates_a_strided_arange, "tl.load(x", "tl.store(x"),
        # Values that leave their position: a transpose, and a sum that may
        # combine another thread's copy of the row.
        (_transposes_its_own_tile, "tl.load(x", "tl.store(x"),
        (_updates_and_sums_what_it_read, "tl.load(x", "tl.store(x"),
        # An atomic reads its target before a later store.
        (_resets_what_it_added_into, "tl.atomic_add(x", "tl.store(x"),
        (_resets_what_an_atomic_returned, "tl.atomic_add(x", "tl.store(x"),
        # A packed load reads anywhere in its tensor before a store; a store
        # lands before a packed load of its tensor.
        (_overwrites_what_a_packed_load_read, "tl.load(x.to(", "tl.store(x"),
        (_packed_load_after_a_store, "tl.store(x", "tl.load(x.to("),
    ],
)
def test_a_write_after_a_read_of_its_tensor_follows_a_barrier(
    fn: object, read: str, write: str
) -> None:
    # The write may overwrite elements another thread has yet to read: one
    # barrier, between the read and the write.
    code = _code(fn)
    assert code.count(_TRITON_BARRIER) == 1, code
    barrier = code.index(_TRITON_BARRIER)
    assert code.index(read) < barrier < code.index(write, barrier), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn",
    [
        _updates_its_own_tile,
        _scales_its_rows_in_a_loop,
        _doubles_its_arange_columns,
        _adds_its_own_columns,
        _updates_an_offset_tile,
    ],
)
def test_an_update_of_its_own_read_takes_no_barrier(fn: object) -> None:
    # ``x[t] = f(x[t])``: every thread writes back only what it read, also
    # in a loop that gets the row's tile from its parent, at an
    # ``hl.arange`` index, and through an atomic.
    assert _TRITON_BARRIER not in _code(fn)


@skipUnlessTritonBarriers
def test_a_write_after_loads_alone_takes_a_bare_barrier() -> None:
    # bar.sync completes every thread's loads before a store, an atomic or a
    # snippet, also between sibling loops: no TMA store needs draining, and
    # only the proxy fence follows, for a TMA write.  A read after a write
    # needs this thread's TMA stores drained, and so do a release, and a
    # write after an atomic's read, which may be a TMA reduction, or after a
    # snippet's, also in an earlier sibling loop.
    with patch(
        "helion._compiler.device_function._selects_tma_access", return_value=True
    ):
        bare = [
            _code(fn)
            for fn in (
                _overwrites_what_it_read,
                _adds_into_what_it_read,
                _snippet_overwrites_what_it_read,
                _sibling_loop_overwrites,
            )
        ]
        full = [
            _code(fn)
            for fn in (
                _reads_what_it_overwrote,
                _releases_into_what_it_read,
                _resets_what_it_added_into,
                _overwrites_what_a_snippet_read,
                _sibling_loop_overwrites_an_atomic_target,
            )
        ]
    for code in bare:
        assert code.count(_TRITON_BARRIER) == 1, code
        assert "async_store_drain" not in code, code
        assert code.index(_TRITON_BARRIER) < code.index("async_load_fence"), code
    for code in full:
        assert code.count(_TRITON_BARRIER) == 1, code
        assert code.index("async_store_drain") < code.index(_TRITON_BARRIER), code


@skipUnlessTritonBarriers
def test_a_bare_barrier_leaves_the_writes_pending() -> None:
    # The store's bare barrier orders no write: the load after the store
    # still takes the full one.
    with patch(
        "helion._compiler.device_function._selects_tma_access", return_value=True
    ):
        code = _code(_rereads_what_it_overwrote)
    assert code.count(_TRITON_BARRIER) == 2, code
    first = code.index(_TRITON_BARRIER)
    second = code.index(_TRITON_BARRIER, first + 1)
    assert first < code.index("tl.store(x") < second, code
    assert first < code.index("async_store_drain") < second, code


@skipUnlessTritonBarriers
def test_a_bare_barrier_leaves_an_atomic_pending() -> None:
    with patch(
        "helion._compiler.device_function._selects_tma_access", return_value=True
    ):
        code = _code(_resets_an_atomic_target_after_a_bare_barrier)
    assert code.count(_TRITON_BARRIER) == 2, code
    first = code.index(_TRITON_BARRIER)
    second = code.index(_TRITON_BARRIER, first + 1)
    assert first < code.index("tl.store(y") < second < code.index("tl.store(x"), code
    assert first < code.index("async_store_drain") < second, code


@skipUnlessTritonBarriers
def test_an_update_under_dynamic_shapes_follows_a_barrier() -> None:
    # The cache key leaves strides out: the binding may run on an expanded
    # view, whose rows share their elements.
    kernel = helion.kernel(_updates_its_own_tile, static_shapes=False)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.randn(32, 128),))
        code = bound.to_code(helion.Config(block_sizes=[1]))
    assert code.count(_TRITON_BARRIER) == 1, code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn", [_updates_its_own_block, _updates_its_own_blocks_in_a_loop]
)
def test_an_update_split_by_epilogue_subtiling_takes_no_barrier(fn: object) -> None:
    # The pieces keep the record of the loads the whole store writes back,
    # and of its index, which tells a loop's iterations apart.
    kernel = helion.kernel(fn, static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.randn(128, 128),))
        code = bound.to_code(
            helion.Config(
                block_sizes=[64, 64], epilogue_subtile=2, indexing="tensor_descriptor"
            )
        )
    assert code.count(".store(") >= 2, code
    assert _TRITON_BARRIER not in code, code


@skipUnlessTritonBarriers
def test_pieces_of_stores_in_two_graphs_follow_a_barrier() -> None:
    # Subtile group ids repeat across graphs: the store after the ``if``
    # overwrites the inner loop's, not pieces of its own.
    kernel = helion.kernel(_overwrites_a_store_of_another_graph, static_shapes=True)
    args = (torch.randn(64, 64), torch.empty(64, 64), torch.ones(1))
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        code = bound.to_code(
            helion.Config(block_sizes=[16, 32], epilogue_subtile=2, indexing="pointer")
        )
    body = _kernel_body(code)
    (branch,) = (statement for statement in body if isinstance(statement, ast.If))
    after = body[body.index(branch) + 1 :]
    stores = [i for i, s in enumerate(after) if "tl.store(out" in ast.unparse(s)]
    assert any(_is_barrier(s) for s in after[: stores[0]]), code


@skipUnlessTritonBarriers
def test_an_update_of_a_transposed_argument_takes_no_barrier() -> None:
    # Distinct elements of the transpose still have distinct addresses.
    code = _code(_updates_its_own_tile, (torch.randn(128, 32).t(),))
    assert _TRITON_BARRIER not in code, code


@skipUnlessTritonBarriers
def test_an_update_through_a_broadcast_view_follows_a_barrier() -> None:
    # Every row of the view is the same memory.
    code = _code(_updates_its_own_tile, (torch.randn(1, 128).expand(32, 128),))
    assert code.count(_TRITON_BARRIER) == 1, code


@skipUnlessTritonBarriers
def test_a_sibling_loop_overwriting_what_an_earlier_one_read_follows_a_barrier() -> (
    None
):
    # One barrier between the loops, not one per iteration of the second.
    body = _kernel_body(_code(_sibling_loop_overwrites))
    loops = [statement for statement in body if isinstance(statement, ast.For)]
    assert len(loops) == 2
    second = body.index(loops[1])
    assert _is_barrier(body[second - 1]), ast.unparse(body[second - 1])
    assert not any(_is_barrier(statement) for statement in ast.walk(loops[1]))


@skipUnlessTritonBarriers
def test_a_write_in_a_loop_after_a_read_before_it_follows_a_barrier() -> None:
    code = _code(_reads_before_the_loop_it_overwrites_in)
    assert code.count(_TRITON_BARRIER) == 1, code
    (loop,) = (
        statement for statement in _kernel_body(code) if isinstance(statement, ast.For)
    )
    assert any(_is_barrier(statement) for statement in loop.body), code


@skipUnlessTritonBarriers
def test_a_loop_overwriting_what_it_read_is_not_warp_specialized() -> None:
    # Warp specialization would split the loop's barrier per partition
    # (``intra_loop_raw_reread_block_ids``): the loop that overwrites what it
    # read keeps its slot pinned to None, the update of its own read does not.
    for fn, pinned in (
        (_overwrites_in_a_loop, True),
        (_updates_in_a_loop, False),
        (_scales_its_rows_in_a_loop, False),
    ):
        with (
            cpu_render_target(),
            patch("helion.language.loops._supports_warp_specialize", return_value=True),
        ):
            specs = _bind(fn).config_spec.range_warp_specialize
        # The row loop's region holds the inner loop.
        assert [spec.pinned for spec in specs] == [pinned, pinned], fn


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (_restores_part_of_what_it_stored, ()),
        (_stores_twice, ()),
        (_stores_wrapping_halves, ()),
        (_stores_a_column_at_its_negative_index, ()),
        (_stores_columns_a_wrap_apart, ()),
        # A dimension of one element takes both halves' indices to it.
        (_stores_the_halves_of_a_row, (torch.randn(32, 1),)),
    ],
)
def test_a_store_over_another_threads_store_follows_a_barrier(
    fn: object, args: tuple[object, ...]
) -> None:
    # The earlier store may land last: one barrier, between the stores.  At
    # the same index too, where the two stores may lay out their values
    # over the threads differently, and at indices narrow enough to wrap.
    code = _code(fn, args)
    assert code.count(_TRITON_BARRIER) == 1, code
    first = code.index("tl.store(x")
    barrier = code.index(_TRITON_BARRIER)
    assert first < barrier < code.index("tl.store(x", barrier), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn",
    [
        _stores_the_halves_of_a_row,
        _stores_sliced_halves,
        _stores_two_slots,
        _stores_the_halves_of_a_tile,
        _stores_broadcast_halves_of_a_tile,
        _stores_rows_a_half_apart,
        _stores_the_halves_of_gathered_rows,
        # A store after a read of the other half.
        _doubles_one_half_into_the_other,
        _doubles_one_arange_half_into_the_other,
    ],
)
def test_accesses_of_disjoint_elements_take_no_barrier(fn: object) -> None:
    # Indices that never meet on one dimension: halves at ``hl.arange``
    # offsets or slices, fixed slots, a tile and that plus its loop's
    # extent, broadcast or not, rows a tile's ``begin`` and that plus a
    # constant, and halves of loaded rows.
    assert _TRITON_BARRIER not in _code(fn)


@skipUnlessTritonBarriers
def test_a_sibling_loop_restoring_what_an_earlier_one_stored_follows_a_barrier() -> (
    None
):
    # One barrier between the loops, not one per iteration of the second.
    body = _kernel_body(_code(_sibling_loop_restores))
    loops = [statement for statement in body if isinstance(statement, ast.For)]
    assert len(loops) == 2
    second = body.index(loops[1])
    assert _is_barrier(body[second - 1]), ast.unparse(body[second - 1])
    assert not any(_is_barrier(statement) for statement in ast.walk(loops[1]))


def _tma_code(fn: object, indexing: list[str] | str, **options: object) -> str:
    """``fn`` of two 128 x 128 tensors, rendered with 64 x 64 blocks (or the
    config ``options``) and ``indexing``, its ``tensor_descriptor`` accesses
    through TMA."""
    with (
        cpu_render_target(),
        patch(
            "helion._compiler.device_function._selects_tma_access",
            return_value=True,
        ),
        # CPU tensors take no TMA; render the descriptor stores anyway.
        patch(
            "helion._compiler.indexing_strategy."
            "TensorDescriptorIndexingStrategy.is_supported",
            return_value=True,
        ),
    ):
        bound = _bind(fn, args=(torch.randn(128, 128), torch.randn(128, 128)))
        return bound.to_code(
            helion.Config(**{"block_sizes": [64, 64], **options}, indexing=indexing)
        )


@skipUnlessTritonBarriers
def test_a_store_over_a_tma_store_drains_it_first() -> None:
    # bar.sync does not complete a TMA store: the barrier before the second
    # store drains the first where it went through TMA, and only there.
    tma_first = _tma_code(_copies_twice, ["pointer", "tensor_descriptor", "pointer"])
    assert tma_first.count(_TRITON_BARRIER) == 1, tma_first
    drain = tma_first.index("async_store_drain")
    assert tma_first.index("x_desc.store") < drain < tma_first.index(_TRITON_BARRIER)
    plain_first = _tma_code(_copies_twice, ["pointer", "pointer", "tensor_descriptor"])
    assert plain_first.count(_TRITON_BARRIER) == 1, plain_first
    assert "async_store_drain" not in plain_first, plain_first


@skipUnlessTritonBarriers
def test_a_store_over_an_earlier_iterations_tma_store_drains_it() -> None:
    # The barrier before the first store is emitted before the second store,
    # of the iteration before, lowers (to TMA here): it drains it.
    code = _tma_code(
        _stores_overlapping_rows_each_iteration,
        ["pointer", "tensor_descriptor", "pointer", "tensor_descriptor"],
    )
    barrier = code.index(_TRITON_BARRIER)
    assert code.index("async_store_drain") < barrier < code.index("x_desc.store")


@skipUnlessTritonBarriers
def test_a_store_restoring_its_own_tile_takes_no_cta_barrier() -> None:
    # The same threads store the same elements each iteration: a TMA store
    # drains the one before, a pointer store needs nothing, and the loop
    # stays open to warp specialization.
    tma = _tma_code(_stores_its_rows_each_iteration, ["pointer", "tensor_descriptor"])
    assert _TRITON_BARRIER not in tma, tma
    assert tma.index("async_store_drain") < tma.index("x_desc.store"), tma
    pointer = _tma_code(_stores_its_rows_each_iteration, ["pointer", "pointer"])
    assert _TRITON_BARRIER not in pointer, pointer
    assert "async_store_drain" not in pointer, pointer
    with (
        cpu_render_target(),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
    ):
        bound = _bind(
            _stores_its_rows_each_iteration,
            args=(torch.randn(128, 128), torch.randn(128, 128)),
        )
        specs = bound.config_spec.range_warp_specialize
    assert specs and not any(spec.pinned for spec in specs)


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "indexing", "block_sizes", "specialized"),
    [
        # The store would drain its own copy from the iteration before.
        (
            _stores_its_rows_each_iteration,
            ["pointer", "tensor_descriptor"],
            [64, 64],
            [None, True],
        ),
        # The barrier before the loop's store would drain the row's.
        (
            _stores_a_row_then_its_tiles,
            ["tensor_descriptor", "pointer", "tensor_descriptor"],
            [64, 64],
            [None, True],
        ),
        # The barrier before the second loop would drain the first's store.
        (
            _stores_its_tiles_in_two_loops,
            ["pointer", "tensor_descriptor", "pointer", "pointer"],
            [16, 64, 64],
            [None, True],
        ),
        # The barrier would drain a store later in the body, not lowered yet,
        # beside another loop's warp-specialized TMA store.
        (
            _restores_rows_beside_another_loop,
            ["pointer"] * 5 + ["tensor_descriptor"],
            [64, 64, 64],
            [None, None, True],
        ),
    ],
)
def test_a_store_a_barrier_drains_goes_through_pointers_beside_warp_specialization(
    fn: object,
    indexing: list[str],
    block_sizes: list[int],
    specialized: list[bool | None],
) -> None:
    # A saved config warp-specializing a loop keeps compiling: a drain
    # beside a warp-specialized TMA store is an InvalidConfig, so the store
    # a barrier would drain goes through pointers and needs none.
    with patch("helion.language.loops._supports_warp_specialize", return_value=True):
        code = _tma_code(
            fn, indexing, block_sizes=block_sizes, range_warp_specializes=specialized
        )
    assert "warp_specialize=True" in code, code
    assert "tl.store(x" in code and "async_store_drain" not in code, code


@skipUnlessTritonBarriers
def test_a_store_a_full_barrier_orders_keeps_tma_beside_warp_specialization() -> None:
    # A full barrier drains every TMA store anyway: the store it orders
    # keeps its TMA lowering, as on main.
    with patch("helion.language.loops._supports_warp_specialize", return_value=True):
        code = _tma_code(
            _stores_and_reloads_a_row_beside_a_loop,
            [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            range_warp_specializes=[None, True],
        )
    assert "warp_specialize=True" in code and "x_desc.store" in code, code


@skipUnlessTritonBarriers
def test_a_full_barrier_beside_a_warp_specialized_tma_store_is_invalid() -> None:
    # A full barrier drains every TMA store of its thread, which a
    # warp-specialized loop's are not: the config is invalid, as main's
    # read-after-write barriers made it.
    with (
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        pytest.raises(helion.exc.InvalidConfig, match="cannot be drained"),
    ):
        _tma_code(
            _stores_then_rereads_its_rows,
            "tensor_descriptor",
            block_sizes=[16, 64, 64],
            range_warp_specializes=[None, True, None],
        )


@skipUnlessTritonBarriers
def test_a_rolled_loops_store_of_its_chunk_takes_no_drain() -> None:
    # Each iteration of the rolled loop stores another chunk of the row.
    code = _tma_code(
        _normalizes_its_rows,
        "tensor_descriptor",
        block_sizes=[16],
        reduction_loops=[64],
    )
    assert "for roffset_" in code and "out_desc.store" in code, code
    assert "async_store_drain" not in code, code
    assert _TRITON_BARRIER not in code, code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn",
    [
        # The loop carries the index, or the index names a column twice.
        _stores_reversed_every_other_iteration,
        _stores_repeating_columns_each_iteration,
    ],
)
def test_a_store_over_another_threads_earlier_store_follows_a_barrier(
    fn: object,
) -> None:
    code = _code(fn, (torch.randn(32, 64), torch.randn(32, 128)))
    assert code.index(_TRITON_BARRIER) < code.index("tl.store(x"), code


@skipIfFn(
    lambda: _get_backend() != "triton" or not gluon_api_available(),
    "the gluon backend runs in the triton lane on Triton >= 3.8",
)
@skipIfRefEager("ref-eager mode generates no code")
def test_gluon_orders_a_write_after_a_read() -> None:
    with gluon_cpu_codegen():
        bound = _bind(_overwrites_what_it_read, backend="gluon")
        code = bound.to_triton_code(helion.Config(block_sizes=[1]))
    barrier = code.index("barrier()")
    assert code.index("load(") < barrier < code.index("store(", barrier), code


@skipUnlessTritonBarriers
@pytest.mark.cuda
def test_a_write_after_a_read_runs_correctly_across_warps() -> None:
    # Each warp reads half a row other warps overwrite with zeros.
    kernel = helion.kernel(
        _overwrites_what_it_read,
        config=helion.Config(block_sizes=[1], num_warps=8),
    )
    x = torch.randn(4096, 128, device=DEVICE)
    torch.testing.assert_close(kernel(x.clone()), x[:, 64:])


@skipUnlessTritonBarriers
@pytest.mark.cuda
def test_a_store_over_another_threads_store_runs_correctly_across_warps() -> None:
    # Each warp stores twos over ones other warps stored.
    kernel = helion.kernel(
        _restores_part_of_what_it_stored,
        config=helion.Config(block_sizes=[1], num_warps=8),
    )
    x = kernel(torch.empty(4096, 128, device=DEVICE))
    expected = torch.ones_like(x)
    expected[:, 64:] = 2.0
    torch.testing.assert_close(x, expected)
