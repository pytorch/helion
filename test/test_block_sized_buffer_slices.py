"""Full slices spanning a block's extent: ``tile.index[:, None]`` keeps
its tile's block, and a host buffer sized by a registered block size keeps a
reduction dim of its own, which CuTe refuses where it meets a loop over that
block (``helion/_compiler/cute/active_blocks.py``,
``reject_tile_alias_reductions``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


@onlyBackends(["triton", "cute"])
@skipIfRefEager("Compiler block sizes are not allocated in ref eager mode")
def test_tile_index_full_slice_keeps_tile_block() -> None:
    @helion.kernel(static_shapes=True)
    def scatter_slice(
        x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        heads, n = x.shape
        for tile_h, tile_n in hl.tile([heads, n]):
            hl.atomic_add(
                out, [idx[tile_n][None, :], tile_h.index[:, None]], x[tile_h, tile_n]
            )
        return out

    @helion.kernel(static_shapes=True)
    def scatter_unsqueeze(
        x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        heads, n = x.shape
        for tile_h, tile_n in hl.tile([heads, n]):
            hl.atomic_add(
                out,
                [idx[tile_n].unsqueeze(0), tile_h.index.unsqueeze(1)],
                x[tile_h, tile_n],
            )
        return out

    # ``tile.index[:, None]`` is ``tile.index.unsqueeze(1)``: the full slice
    # keeps the tile's block.  A reduction dim allocated for it is indexed by
    # nothing, yet CuTe put it on a thread axis whose threads all repeated the
    # tile's work (and only the first issued the atomic).
    args = (torch.randn(8, 64), torch.randint(0, 4, (64,)), torch.zeros(4, 8))
    launches = []
    with cpu_render_target():
        for kernel in (scatter_slice, scatter_unsqueeze):
            bound = cpu_bind(kernel, args)
            assert not any(block.reduction for block in bound.env.block_sizes)
            code = bound.to_code(helion.Config(block_sizes=[8, 32]))
            assert "_RDIM_SIZE" not in code
            (launch,) = [line for line in code.splitlines() if "_launcher(" in line]
            launches.append(launch.replace(kernel.fn.__name__, "kernel"))
    assert launches[0] == launches[1]


@onlyBackends(["triton", "cute"])
@skipIfRefEager("Compiler block sizes are not allocated in ref eager mode")
def test_block_sized_buffer_full_slice_keeps_reduction_dim() -> None:
    @helion.kernel(static_shapes=True)
    def double_rows(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        block_n = hl.register_block_size(n)
        buf = torch.ones([m, block_n], dtype=torch.float32, device=x.device)
        out = torch.empty_like(buf)
        for tile_m in hl.tile(m):
            out[tile_m, :] = buf[tile_m, :] * 2
        return out

    # Unlike ``tile.index``, a host buffer sized by a block has a dim no loop
    # covers: tracing allocates the reduction dim that indexes its slice.  One
    # allocated during codegen instead would have no strategy, and its index
    # would span the block's size hint rather than the configured block.
    with cpu_render_target():
        bound = cpu_bind(double_rows, (torch.empty(8, 256),))
        assert any(block.reduction for block in bound.env.block_sizes)
        allocated = len(bound.env.block_sizes)
        bound.to_code(helion.Config(block_sizes=[128, 8]))
    assert len(bound.env.block_sizes) == allocated


def _buffer_slice_beside_grid_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, block_n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + buf[tile_m, :]
    return out


def _buffer_slice_beside_device_loop_tile(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + buf[tile_m, :]
    return out


def _buffer_row_into_device_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _device_loop_row_into_buffer(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_zeros([m, block_n])
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        buf[tile_m, :] = acc
    return buf


def _buffer_row_carried_into_device_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        prev = hl.zeros([tile_m, block_n], dtype=x.dtype)
        last = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for _ in hl.grid(2):
            # The row reaches ``last`` in the second iteration.
            last = prev
            prev = row
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + last
    return out


def _buffer_zeroed_under_a_device_loop_mask(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
        for tile_n in hl.tile(n, block_size=block_n):
            mask = x[tile_m, tile_n] > 0
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
        if mask:
            buf[tile_m, :] = 0
    return out, buf


def _buffer_row_reshaped_into_device_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        # Flattened, the row's dims are one product of both blocks.
        row = (row.reshape(-1) + 1).reshape([tile_m, block_n])
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _device_loop_under_a_buffer_mask(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        mask = buf[tile_m, :] > 0
        if mask:
            for tile_n in hl.tile(n, block_size=block_n):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out


def _buffer_slice_in_a_constant_branch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, block_n]):
        if block_n > 64:
            out[tile_m, tile_n] = x[tile_m, tile_n]
        else:
            out[tile_m, tile_n] = x[tile_m, tile_n] + buf[tile_m, :]
    return out


def _buffer_zeroed_before_device_loop(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        buf[tile_m, :] = 0
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out, buf


def _buffer_doubled_before_device_loop(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    doubled = torch.empty_like(buf)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        doubled[tile_m, :] = buf[tile_m, :] * 2
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out, doubled


def _buffer_row_carried_beside_device_loop_value(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    doubled = torch.empty_like(buf)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        scale = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for _ in hl.grid(2):
            acc = acc + row
            scale = scale + 1
        doubled[tile_m, :] = acc
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out, doubled


def _buffer_zeroed_in_a_constant_branch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        if block_n <= 64:
            buf[tile_m, :] = 0
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out, buf


def _buffer_zeroed_beside_a_branch_value(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        bias = hl.zeros([tile_m, block_n], dtype=x.dtype)
        if flags[0] > 0:
            buf[tile_m, :] = 0
            bias = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + bias
    return out, buf


def _device_loop_row_doubled_beside_zeroing_the_buffer(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        if flags[0] > 0:
            buf[tile_m, :] = 0
            acc = acc * 2
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + acc
    return out, buf


def _buffer_row_out_of_a_branch_into_device_loop(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = hl.zeros([tile_m, block_n], dtype=x.dtype)
        if flags[0] > 0:
            row = buf[tile_m, :] * 2
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _device_loop_row_stored_in_a_branch(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_zeros([m, block_n])
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        if flags[0] > 0:
            buf[tile_m, :] = acc
    return buf


def _buffer_slice_on_a_constant_branch_inside_device_loop(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        # Zeroing the buffer keeps its reduction block live at every config.
        buf[tile_m, :] = 0
        for tile_n in hl.tile(n, block_size=block_n):
            if block_n <= 64:
                out[tile_m, tile_n] = x[tile_m, tile_n] + buf[tile_m, :]
            else:
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2
    return out, buf


def _buffer_row_carried_beside_device_loop_value_by_a_while(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    doubled = torch.empty_like(buf)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        scale = hl.zeros([tile_m, block_n], dtype=x.dtype)
        count = flags[0]
        while count > 0:
            acc = acc + row
            scale = scale + 1
            count = count - 1
        doubled[tile_m, :] = acc
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] * scale
    return out, doubled


def _buffer_row_carried_into_device_loop_by_a_while(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        prev = hl.zeros([tile_m, block_n], dtype=x.dtype)
        last = hl.zeros([tile_m, block_n], dtype=x.dtype)
        count = flags[0]
        while count > 0:
            # The row reaches ``last`` in the second iteration.
            last = prev
            prev = row
            count = count - 1
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + last
    return out


def _buffer_row_replaced_on_a_constant_branch(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        if block_n <= 64:
            row = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _buffer_row_overwritten_in_device_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        for tile_n in hl.tile(n, block_size=block_n):
            row = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _buffer_row_overwritten_in_a_branch_of_device_loop(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        for tile_n in hl.tile(n, block_size=block_n):
            if flags[0] > 0:
                row = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
            else:
                row = hl.full([tile_m, block_n], 3.0, dtype=x.dtype)
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _buffer_row_kept_by_a_branch_of_device_loop(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        for tile_n in hl.tile(n, block_size=block_n):
            # The other side passes the row through.
            if flags[0] > 0:
                row = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _buffer_row_chosen_by_a_device_loop_mask(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
        for tile_n in hl.tile(n, block_size=block_n):
            mask = x[tile_m, tile_n] > 0
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
        row = hl.zeros([tile_m, block_n], dtype=x.dtype)
        if mask:
            row = hl.full([tile_m, block_n], 5.0, dtype=x.dtype)
        buf[tile_m, :] = row
    return out, buf


def _device_loop_row_chosen_by_a_buffer_mask(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = hl.zeros([tile_m, block_n], dtype=x.dtype)
        if buf[tile_m, :] > 0:
            row = hl.full([tile_m, block_n], 2.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + row
    return out


def _buffer_zeroed_while_a_device_loop_mask_holds(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
        for tile_n in hl.tile(n, block_size=block_n):
            mask = x[tile_m, tile_n] > 0
            out[tile_m, tile_n] = x[tile_m, tile_n] * 2
        while mask:
            buf[tile_m, :] = 0
            mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
    return out, buf


def _device_loop_run_while_a_buffer_mask_holds(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        mask = buf[tile_m, :] > 0
        while mask:
            for tile_n in hl.tile(n, block_size=block_n):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2
            mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
    return out


def _device_loop_row_times_the_buffer_row(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = x.new_empty([m])
    for tile_m in hl.tile(m):
        row = buf[tile_m, :]
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = (acc * row).sum(-1)
    return out


def _device_loop_row_times_the_buffer_row_read_after_it(
    x: torch.Tensor,
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = x.new_empty([m])
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = (acc * buf[tile_m, :]).sum(-1)
    return out


def _device_loop_row_selected_by_the_buffer_row(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = x.new_empty([m])
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = torch.where(buf[tile_m, :] > 0, acc, 0.0).amax(-1)
    return out


def _buffer_row_doubled_while_a_device_loop_mask_holds(
    x: torch.Tensor,
) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n])
    out = x.new_empty([m])
    for tile_m in hl.tile(m):
        mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
        for tile_n in hl.tile(n, block_size=block_n):
            mask = x[tile_m, tile_n] > 0
        row = buf[tile_m, :]
        while mask:
            row = row * 2
            mask = hl.zeros([tile_m, block_n], dtype=torch.bool)
        out[tile_m] = row.sum(-1)
    return out


_INSIDE = "is accessed inside that loop"
_THROUGH_A_VALUE = "meets that loop through a value carrying their shared dim"


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    ("fn", "dtype", "meeting"),
    # Each case is one way the slice meets the tile (no product of axes);
    # the dtypes are the ones whose lowering differed.
    [
        # int32, int64 and float64 loads skip the vector-load path, where the
        # refusal of the grid case used to sit.
        (_buffer_slice_beside_grid_tile, torch.float32, _INSIDE),
        (_buffer_slice_beside_grid_tile, torch.int32, _INSIDE),
        (_buffer_slice_beside_grid_tile, torch.int64, _INSIDE),
        (_buffer_slice_beside_grid_tile, torch.float64, _INSIDE),
        (_buffer_slice_beside_device_loop_tile, torch.float32, _INSIDE),
        (_buffer_slice_beside_device_loop_tile, torch.int32, _INSIDE),
        (_buffer_row_into_device_loop, torch.float32, _THROUGH_A_VALUE),
        (_device_loop_row_into_buffer, torch.float32, _THROUGH_A_VALUE),
        (_buffer_row_carried_into_device_loop, torch.float32, _THROUGH_A_VALUE),
        (_buffer_row_reshaped_into_device_loop, torch.float32, _THROUGH_A_VALUE),
        (_buffer_zeroed_under_a_device_loop_mask, torch.float32, _THROUGH_A_VALUE),
        (_device_loop_under_a_buffer_mask, torch.float32, _THROUGH_A_VALUE),
        (_buffer_row_chosen_by_a_device_loop_mask, torch.float32, _THROUGH_A_VALUE),
        (_device_loop_row_chosen_by_a_buffer_mask, torch.float32, _THROUGH_A_VALUE),
        (
            _buffer_zeroed_while_a_device_loop_mask_holds,
            torch.float32,
            _THROUGH_A_VALUE,
        ),
        (_device_loop_run_while_a_buffer_mask_holds, torch.float32, _THROUGH_A_VALUE),
    ],
)
def test_cute_refuses_a_block_sized_buffer_slice_beside_its_tile(
    fn: Callable[[torch.Tensor], object], dtype: torch.dtype, meeting: str
) -> None:
    """``buf[tile_m, :]`` of a host buffer sized by the block a loop tiles:
    the slice's reduction dim reuses ``tile_n``'s variable, so the two dims
    pair element by element (Triton's ``tl.arange`` lanes do).  CuTe read the
    grid tile's index there, past the buffer's first ``block_n`` columns,
    and beside a device-loop tile gave the slice lanes of its own, each
    storing a different buffer column into the same element of ``out``,
    also when the row is read before the loop.  It refuses each kernel at
    codegen, for every dtype, and so it does when a row accumulated over the
    tile is stored into the slice after the loop, and when another loop
    carries the row into the value the tile's loop reads, one iteration
    later.  So it does for a row flattened and reshaped back before the
    loop, whose flat dim is a product of both blocks.  A tensor ``if`` pairs
    its test with what it decides element by element, so it is refused when
    the loop's last mask decides where the buffer is zeroed (CuTe rendered
    each thread's one mask element deciding every buffer column the thread
    stores), and when the buffer decides where the loop runs; so it is when
    the mask picks the row stored into the buffer, or the buffer the row the
    loop reads, and when a ``while`` condition takes the test's place."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 64, dtype=dtype),))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match=f"{meeting} .*reuses tile block 0's variable",
        ):
            bound.to_code(helion.Config(block_sizes=[32, 8]))


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    "fn",
    [
        _buffer_zeroed_before_device_loop,
        _buffer_doubled_before_device_loop,
        _buffer_row_carried_beside_device_loop_value,
        _buffer_zeroed_in_a_constant_branch,
        _buffer_row_overwritten_in_device_loop,
    ],
)
def test_cute_renders_a_block_sized_buffer_slice_apart_from_its_tile(
    fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """The same slice before the tile's loop, connected to it by no value:
    zeroing the buffer, or doubling it into another such buffer.  The tile
    has no coordinates there and the slice keeps its own, so CuTe renders the
    kernel, as it did before it refused the slice beside the tile.  So it
    does when another loop carries the row beside a value the tile's loop
    reads (a loop's result carries only what its body computes it from), on
    the taken side of a branch constant for the config, and when the tile's
    loop overwrites the row before it reads it (reading only its shape)."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 64),))
        bound.to_code(helion.Config(block_sizes=[32, 8]))


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    ("fn", "refused"),
    [
        (_buffer_row_overwritten_in_a_branch_of_device_loop, False),
        (_buffer_row_kept_by_a_branch_of_device_loop, True),
    ],
)
def test_cute_follows_a_row_through_the_branches_in_the_loop(
    fn: Callable[..., object], refused: bool
) -> None:
    """The row read before the tile's loop, which each branch of a runtime
    ``if`` in the loop overwrites before the loop reads it: the branches
    receive a copy of the row and read only its shape, so the slice meets
    the loop nowhere and CuTe renders the kernel, as it did before it
    refused the slice beside the tile.  Overwritten in one branch only, the
    row reaches the loop through the other side, which passes it through
    (CuTe rendered every lane adding a different buffer column into one
    element when the flag is 0), and the kernel is refused."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 64), torch.ones(1, dtype=torch.int32)))
        config = helion.Config(block_sizes=[32, 8])
        if refused:
            with pytest.raises(
                helion.exc.BackendUnsupported,
                match=f"{_THROUGH_A_VALUE} .*reuses tile block 0's variable",
            ):
                bound.to_code(config)
        else:
            bound.to_code(config)


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    "fn",
    [
        _device_loop_row_times_the_buffer_row,
        _device_loop_row_times_the_buffer_row_read_after_it,
        _device_loop_row_selected_by_the_buffer_row,
        _buffer_row_doubled_while_a_device_loop_mask_holds,
    ],
)
def test_cute_refuses_the_slice_and_the_loop_combined_after_it(
    fn: Callable[[torch.Tensor], object],
) -> None:
    """A row the tile's loop accumulates and the slice's row, combined after
    the loop: their elements pair by position, as they would inside it.
    CuTe rendered each of the loop's threads storing its one element times
    the sum of the buffer row into the same output element, so the kernel
    is refused, whether the slice is read before or after the loop, and
    when a ``while`` condition the loop computes decides how often each
    element of the row is doubled."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 64),))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match=f"{_THROUGH_A_VALUE} .*reuses tile block 0's variable",
        ):
            bound.to_code(helion.Config(block_sizes=[8, 8]))


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    ("fn", "refused"),
    [
        (_buffer_zeroed_beside_a_branch_value, False),
        (_device_loop_row_doubled_beside_zeroing_the_buffer, False),
        (_buffer_row_out_of_a_branch_into_device_loop, True),
        (_device_loop_row_stored_in_a_branch, True),
    ],
)
def test_cute_follows_values_through_the_branch_holding_the_slice(
    fn: Callable[..., object], refused: bool
) -> None:
    """A branch holding the slice connects it with the tile's loop only
    through the results it computes from the slice, or the values it stores
    into it: a bias the branch sets beside zeroing the buffer, or the loop's
    row it doubles there, meets the slice nowhere, and CuTe renders both
    kernels, while a row the branch reads from the buffer into the loop, or
    the loop's row it stores into the buffer, is refused."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 64), torch.ones(1, dtype=torch.int32)))
        config = helion.Config(block_sizes=[32, 8])
        if refused:
            with pytest.raises(
                helion.exc.BackendUnsupported,
                match=f"{_THROUGH_A_VALUE} .*reuses tile block 0's variable",
            ):
                bound.to_code(config)
        else:
            bound.to_code(config)


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    "fn",
    [
        _buffer_slice_in_a_constant_branch,
        _buffer_slice_on_a_constant_branch_inside_device_loop,
    ],
)
def test_cute_refuses_a_block_sized_buffer_slice_a_constant_branch_takes(
    fn: Callable[[torch.Tensor], object],
) -> None:
    """The slice beside the tile on the taken side of a branch constant for
    the config is refused as it is without the branch: beside the grid tile
    its reduction block used to drop out of the taken side's live blocks, so
    nothing refused it and codegen crashed looking up its strategy.  On the
    untaken side the slice is never emitted, and the kernel renders, also
    when zeroing the buffer before the device loop keeps its reduction block
    live (the untaken branch's access used to be refused)."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 256, dtype=torch.int32),))
        bound.to_code(helion.Config(block_sizes=[128, 8]))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match=f"{_INSIDE} .*reuses tile block 0's variable",
        ):
            bound.to_code(helion.Config(block_sizes=[32, 8]))


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
def test_cute_skips_the_untaken_side_of_a_constant_branch() -> None:
    """The row a constant branch replaces reaches the tile's loop only on
    the side the config takes: replaced (``block_n`` 32), the slice's row
    passes through the untaken side alone and the kernel renders; kept
    (``block_n`` 128), it reaches the loop and is refused."""
    kernel = helion.kernel(
        _buffer_row_replaced_on_a_constant_branch, backend="cute", static_shapes=True
    )
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.zeros(8, 256),))
        bound.to_code(helion.Config(block_sizes=[32, 8]))
        with pytest.raises(
            helion.exc.BackendUnsupported,
            match=f"{_THROUGH_A_VALUE} .*reuses tile block 0's variable",
        ):
            bound.to_code(helion.Config(block_sizes=[128, 8]))


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of a block-sized buffer slice")
@pytest.mark.parametrize(
    ("fn", "refused"),
    [
        (_buffer_row_carried_beside_device_loop_value_by_a_while, False),
        (_buffer_row_carried_into_device_loop_by_a_while, True),
    ],
)
def test_cute_follows_values_through_a_while_loop_beside_the_slice(
    fn: Callable[..., object], refused: bool
) -> None:
    """A while loop carries each value it writes through its iterations, as
    a for loop does, and nothing else: a row it accumulates beside a count
    the tile's loop reads meets the loop nowhere and renders, while a row it
    carries into a value the loop reads is refused."""
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(
            kernel, (torch.zeros(8, 64), torch.full((1,), 2, dtype=torch.int32))
        )
        config = helion.Config(block_sizes=[32, 8])
        if refused:
            with pytest.raises(
                helion.exc.BackendUnsupported,
                match=f"{_THROUGH_A_VALUE} .*reuses tile block 0's variable",
            ):
                bound.to_code(config)
        else:
            bound.to_code(config)


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@skipIfRefEager("hl.register_block_size returns the whole dim in ref eager mode")
def test_slices_a_constant_branch_or_a_while_keeps_apart_match_eager() -> None:
    x = torch.randn(8, 256, device=DEVICE)
    kernel = helion.kernel(
        _buffer_slice_on_a_constant_branch_inside_device_loop,
        static_shapes=True,
        autotune_effort="none",
    )
    _code, (out, buf) = code_and_output(kernel, (x,), block_sizes=[128, 8])
    torch.testing.assert_close(out, x * 2)
    torch.testing.assert_close(buf, torch.zeros(8, 128, device=DEVICE))
    x = torch.randn(8, 48, device=DEVICE)
    flags = torch.full((1,), 3, dtype=torch.int32, device=DEVICE)
    kernel = helion.kernel(
        _buffer_row_carried_beside_device_loop_value_by_a_while,
        static_shapes=True,
        autotune_effort="none",
    )
    _code, (out, doubled) = code_and_output(kernel, (x, flags), block_sizes=[32, 8])
    torch.testing.assert_close(out, x * 3)
    torch.testing.assert_close(doubled, torch.full((8, 32), 3.0, device=DEVICE))
    # The row a constant branch replaces, and one the loop overwrites.
    for fn in (
        _buffer_row_replaced_on_a_constant_branch,
        _buffer_row_overwritten_in_device_loop,
    ):
        kernel = helion.kernel(fn, static_shapes=True, autotune_effort="none")
        _code, out = code_and_output(kernel, (x,), block_sizes=[32, 8])
        torch.testing.assert_close(out, x + 2)
    # The row each branch of a runtime ``if`` in the loop overwrites.
    kernel = helion.kernel(
        _buffer_row_overwritten_in_a_branch_of_device_loop,
        static_shapes=True,
        autotune_effort="none",
    )
    for taken, added in ((0, 3.0), (1, 2.0)):
        flags = torch.full((1,), taken, dtype=torch.int32, device=DEVICE)
        _code, out = code_and_output(kernel, (x, flags), block_sizes=[32, 8])
        torch.testing.assert_close(out, x + added)


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@skipIfRefEager("hl.register_block_size returns the whole dim in ref eager mode")
def test_branch_beside_a_block_sized_buffer_slice_matches_eager() -> None:
    x = torch.randn(8, 48, device=DEVICE)
    # The 48 columns in tiles of 32: the row sums the two tiles' columns, the
    # ragged second one adding its 16, and the second loop adds it to each.
    row = x[:, :32].clone()
    row[:, :16] += x[:, 32:]
    for fn, expected_out in (
        (_buffer_zeroed_beside_a_branch_value, lambda taken: x + 2 * taken),
        (
            _device_loop_row_doubled_beside_zeroing_the_buffer,
            lambda taken: x + torch.cat([row, row[:, :16]], dim=1) * (1 + taken),
        ),
    ):
        kernel = helion.kernel(fn, static_shapes=True, autotune_effort="none")
        for taken in (0, 1):
            flags = torch.full((1,), taken, dtype=torch.int32, device=DEVICE)
            _code, (out, buf) = code_and_output(kernel, (x, flags), block_sizes=[32, 8])
            torch.testing.assert_close(out, expected_out(taken))
            torch.testing.assert_close(
                buf, torch.full((8, 32), 1.0 - taken, device=DEVICE)
            )


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@skipIfRefEager("hl.register_block_size returns the whole dim in ref eager mode")
@pytest.mark.parametrize(
    ("fn", "fill"),
    [
        (_buffer_zeroed_before_device_loop, 0.0),
        (_buffer_doubled_before_device_loop, 2.0),
    ],
)
def test_block_sized_buffer_slice_apart_from_its_tile_matches_eager(
    fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]], fill: float
) -> None:
    kernel = helion.kernel(fn, static_shapes=True, autotune_effort="none")
    x = torch.randn(8, 48, device=DEVICE)
    _code, (out, buf) = code_and_output(kernel, (x,), block_sizes=[32, 8])
    torch.testing.assert_close(out, x * 2)
    torch.testing.assert_close(buf, torch.full((8, 32), fill, device=DEVICE))


def _buffer_columns_by_grid_tile_lane(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n]).cumsum(1)
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, block_n]):
        columns = buf[tile_m, tile_n.index - tile_n.begin]
        out[tile_m, tile_n] = x[tile_m, tile_n] + columns
    return out


def _buffer_columns_by_device_loop_tile_lane(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block_n = hl.register_block_size(n)
    buf = x.new_ones([m, block_n]).cumsum(1)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=block_n):
            columns = buf[tile_m, tile_n.index - tile_n.begin]
            out[tile_m, tile_n] = x[tile_m, tile_n] + columns
    return out


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["triton", "cute"])
@skipIfRefEager("hl.register_block_size returns the whole dim in ref eager mode")
@pytest.mark.parametrize(
    "fn", [_buffer_columns_by_grid_tile_lane, _buffer_columns_by_device_loop_tile_lane]
)
def test_block_sized_buffer_indexed_by_tile_lane(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A host buffer sized by the block a loop tiles, indexed by the tile's
    lane (``tile.index - tile.begin``), which CuTe's refusal of the full slice
    ``buf[tile_m, :]`` beside that tile suggests.
    Every tile, the ragged last one too, adds the buffer's columns ``1..32``
    from its own first column."""
    kernel = helion.kernel(
        fn,
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[exc.TensorOperationInWrapper],
    )
    x = torch.randn(8, 48, device=DEVICE)
    _, actual = code_and_output(kernel, (x,), block_sizes=[32, 8])
    columns = torch.arange(1, 33, device=DEVICE, dtype=x.dtype)
    torch.testing.assert_close(actual, x + torch.cat([columns, columns[:16]]))
