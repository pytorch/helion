"""Per-item DMA-orchestrated jagged reduce template for Pallas/TPU.

Computes a per-item reduction along the second-minor axis of a jagged
tensor::

    out[i, :] = reduce(
        jagged_data[jagged_dim_offsets[i] : jagged_dim_offsets[i + 1], :]
    )

I/O::

    jagged_data        : [L, M_padded]   values, M_padded a multiple of 128
    jagged_dim_offsets : i32[N + 1]      cumulative offsets, [N] == L
    out                : [N, M_padded]   per-item reduction

The ``jagged_tile_size`` argument is the block size of the inner Helion
``hl.tile(...)`` that the dispatch detector pairs with the
``hl.jagged_tile(...)`` over the items axis.

Structure:

  - Single-program grid. Inside the kernel, ``pl.loop`` iterates the
    item index over ``[0, num_items)``.
  - ``jagged_dim_offsets`` is prefetched into SMEM; the per-item
    ``(item_start, item_end, item_length)`` is re-derived inside the
    closures that need it.
  - ``jagged_data`` is fetched into VMEM via ``pltpu.make_async_copy``
    one ``jagged_tile_size``-row block at a time, double-buffered with
    SMEM ping-pong indices.
  - A fp32 per-item accumulator lives in VMEM scratch (``acc_ref``).
    It is zero-initialised when entering each item (``tile_idx == 0``)
    and flushed via a separate double-buffered output DMA at the item
    boundary (``tile_idx == num_tiles - 1``).
  - No cross-program writes — the output write surface is the per-item
    DMA flush.

Padding contract (caller):

  - ``jagged_data.shape[1]`` must be a multiple of 128 (lane alignment).
  - ``jagged_tile_size`` must be a multiple of 8 (sublane alignment).
  - ``jagged_dim_offsets`` must be int32 (TPU/Pallas rejects int64).

Known limitation: an empty item (length 0 → ``num_tiles == 0``) skips
the inner loop entirely, so the accumulator is neither initialised nor
flushed — that item's output row in HBM keeps whatever the caller
pre-filled it with (typically zero from ``torch.zeros``).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _cdiv(x: object, y: object) -> object:
    return (x + y - 1) // y  # type: ignore[operator]


_LANE_ALIGN = 128
_SUBLANE_ALIGN = 8


class _ItemBounds(NamedTuple):
    """Per-item row-range bounds in ``jagged_data``'s L-space."""

    start: jax.Array
    end: jax.Array
    length: jax.Array


def _derive_item_bounds(item_idx: object, jagged_offsets_ref: object) -> _ItemBounds:
    start = jagged_offsets_ref[item_idx]  # type: ignore[index]
    end = jagged_offsets_ref[item_idx + 1]  # type: ignore[index,operator]
    length = end - start
    return _ItemBounds(start=start, end=end, length=length)


def _jagged_reduce_kernel(*args: object, **kwargs: object) -> None:
    """Outer dispatcher: iterate ``item_idx`` over the items axis."""
    jagged_offsets_ref = args[0]
    num_items = jagged_offsets_ref.shape[0] - 1  # type: ignore[attr-defined]

    @pl.loop(0, num_items)
    def _(item_idx: object) -> None:
        _jagged_reduce_per_item(item_idx, *args, **kwargs)  # type: ignore[arg-type]


def _jagged_reduce_per_item(
    item_idx: object,
    # Prefetched in SMEM
    jagged_offsets_ref: object,  # i32[num_items + 1]
    sem_ids_ref: object,  # i32[2] — (data_sem_idx, output_sem_idx) ping-pong
    block_output_owner_ref: object,  # i32[2] — item_idx each output block holds
    # HBM
    jagged_data_ref: object,  # jagged_data.dtype[L, m_padded]
    output_ref: object,  # o_dtype[num_items, m_padded]
    # VMEM scratch (block_* = a buffer whose shape carries a block size)
    block_data_ref: object,  # jagged_data.dtype[2, tile_size, m_padded]
    block_output_ref: object,  # o_dtype[2, m_padded]
    sems: object,  # DMA semaphores[2, 2]
    acc_ref: object,  # fp32[m_padded]
    *,
    tile_size: int,
) -> None:
    o_dtype = block_output_ref.dtype
    _, m_padded = jagged_data_ref.shape
    num_items = jagged_offsets_ref.shape[0] - 1

    bounds = _derive_item_bounds(item_idx, jagged_offsets_ref)

    # ------------------------------------------------------------------ #
    # DMA wrappers                                                        #
    # ------------------------------------------------------------------ #

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_data(item_idx, tile_idx, data_sem_idx, *, wait=False):
        """HBM jagged_data → VMEM block_data ping-pong slot."""
        sem = sems.at[0, data_sem_idx]
        vmem_ref = block_data_ref.at[data_sem_idx]

        b = _derive_item_bounds(item_idx, jagged_offsets_ref)
        row_start = b.start + tile_idx * tile_size
        sz = jnp.minimum(tile_size, b.end - row_start)

        if not wait:
            _async_copy(
                jagged_data_ref.at[pl.ds(row_start, sz), :],
                vmem_ref.at[pl.ds(0, sz), :],
                sem,
                wait=False,
            )
        else:
            # Degenerate copy as a semaphore wait.
            dst = vmem_ref.at[pl.ds(0, sz), :]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _send_output(item_idx, output_sem_idx, *, wait=False):
        """VMEM block_output slot → HBM out[item_idx, :]."""
        sem = sems.at[1, output_sem_idx]
        vmem_ref = block_output_ref.at[output_sem_idx]

        if not wait:
            _async_copy(
                vmem_ref,
                output_ref.at[item_idx, :],
                sem,
                wait=False,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def start_fetch_data(item_idx, tile_idx, data_sem_idx):
        return _fetch_data(item_idx, tile_idx, data_sem_idx)

    def wait_fetch_data(item_idx, tile_idx, data_sem_idx):
        return _fetch_data(item_idx, tile_idx, data_sem_idx, wait=True)

    def start_send_output(item_idx, output_sem_idx):
        # Record which item this buffer is holding so the matching
        # wait_send_output later knows what to drain.
        block_output_owner_ref[output_sem_idx] = item_idx
        _send_output(item_idx, output_sem_idx)

    def wait_send_output(output_sem_idx):
        prev_owner = block_output_owner_ref[output_sem_idx]

        # Skip at cold start (-1 sentinel) and skip if the buffer is
        # already past us.
        @pl.when(jnp.logical_and(prev_owner >= 0, prev_owner <= item_idx))
        def _():
            _send_output(prev_owner, output_sem_idx, wait=True)

    # ------------------------------------------------------------------ #
    # Inner loop: walk tile_size-row blocks of this item's data           #
    # ------------------------------------------------------------------ #

    def process():
        num_tiles = _cdiv(bounds.length, tile_size)

        def next_tile_ids(item_idx, tile_idx, data_sem_idx):
            next_tile = tile_idx + 1
            is_last_tile = next_tile == num_tiles
            next_tile = lax.select(is_last_tile, 0, next_tile)
            next_item = lax.select(is_last_tile, item_idx + 1, item_idx)
            next_data_sem = lax.select(data_sem_idx == 0, 1, 0)
            return next_item, next_tile, next_data_sem

        @pl.loop(0, num_tiles, unroll=False)
        def compute_tile(tile_idx):
            # Init acc at the item boundary.
            @pl.when(tile_idx == 0)
            def init_acc():
                acc_ref[...] = jnp.zeros_like(acc_ref)

            data_sem_idx = sem_ids_ref[0]
            next_item, next_tile, next_data_sem = next_tile_ids(
                item_idx, tile_idx, data_sem_idx
            )

            # Prefetch the next block (advance SMEM ping-pong, then fire).
            @pl.when(next_item < num_items)
            def prefetch_next():
                sem_ids_ref[0] = next_data_sem
                start_fetch_data(next_item, next_tile, next_data_sem)

            # Wait for the current block.
            wait_fetch_data(item_idx, tile_idx, data_sem_idx)

            # Compute: mask partial-tail rows (zero is the sum identity),
            # reduce, accumulate.
            row_start = tile_idx * tile_size
            sz = jnp.minimum(tile_size, bounds.length - row_start)
            data_block = block_data_ref[data_sem_idx]  # (tile_size, m_padded)
            iota = lax.broadcasted_iota(jnp.int32, (tile_size, 1), 0)
            mask = iota < sz
            masked = jnp.where(mask, data_block, jnp.zeros_like(data_block))
            acc_ref[...] = acc_ref[...] + masked.sum(axis=0, dtype=jnp.float32)

            # Flush at the item boundary.
            @pl.when(tile_idx == num_tiles - 1)
            def flush_output():
                # Drain the other output block so we can reuse it,
                # then advance SMEM ping-pong.
                output_sem_idx = sem_ids_ref[1]
                sem_ids_ref[1] = lax.select(output_sem_idx == 0, 1, 0)
                wait_send_output(output_sem_idx)

                # Cast acc → o_dtype, store to the output block,
                # kick off the DMA.
                block_output_ref.at[output_sem_idx][...] = acc_ref[...].astype(o_dtype)
                start_send_output(item_idx, output_sem_idx)

    @pl.when(item_idx == 0)
    def prologue():
        # Cold-start the prefetch chain — the pl.loop-driven prefetch
        # has no "previous" iteration to fire for item_idx == 0.
        start_fetch_data(item_idx=0, tile_idx=0, data_sem_idx=0)

    @pl.when(item_idx < num_items)
    def pipeline():
        process()

    @pl.when(item_idx == num_items - 1)
    def epilogue():
        # Drain both output ping-pong blocks before returning,
        # otherwise the last item's output may never reach HBM.
        for i in range(2):
            wait_send_output(output_sem_idx=i)


def _prepare_inputs(jagged_data: jax.Array, m_actual: int) -> jax.Array:
    m_padded = _align_to(m_actual, _LANE_ALIGN)
    if m_padded != m_actual:
        jagged_data = jnp.pad(
            jagged_data,
            ((0, 0), (0, m_padded - m_actual)),
            constant_values=0,
        )
    return jagged_data


def _prepare_outputs(output_padded: jax.Array, m_actual: int) -> jax.Array:
    return output_padded[:, :m_actual]


def jagged_reduce_pallas(
    jagged_data: jax.Array,  # [L, M_actual]
    jagged_dim_offsets: jax.Array,  # i32[N + 1]
    *,
    o_dtype: object = None,
    jagged_tile_size: int = 64,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:  # [N, M_actual]
    """Jagged reduction over the second-minor axis (sum).

    See module docstring for I/O shapes and the padding contract.
    """
    if o_dtype is None:
        o_dtype = jagged_data.dtype
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    if jagged_tile_size % _SUBLANE_ALIGN != 0:
        raise ValueError(
            f"jagged_tile_size={jagged_tile_size} must be a multiple of "
            f"{_SUBLANE_ALIGN} (TPU sublane alignment)."
        )

    _, m_actual = jagged_data.shape
    num_items = jagged_dim_offsets.shape[0] - 1

    # Lane-pad the trailing dim (zero is the sum identity).
    jagged_data_padded = _prepare_inputs(jagged_data, m_actual)
    _, m_padded = jagged_data_padded.shape

    # SMEM initial state: ping-pong indices start at 0; per-block
    # ownership starts at -1 (sentinel: "block never used").
    init_sem_ids = jnp.zeros((2,), jnp.int32)
    init_block_output_owner = jnp.full((2,), -1, jnp.int32)

    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

    scratch_shapes = [
        pltpu.VMEM((2, jagged_tile_size, m_padded), jagged_data_padded.dtype),
        pltpu.VMEM((2, m_padded), o_dtype),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.VMEM((m_padded,), jnp.float32),
    ]

    scalar_prefetches = (
        jagged_dim_offsets,
        init_sem_ids,
        init_block_output_owner,
    )

    kernel = pl.pallas_call(
        functools.partial(_jagged_reduce_kernel, tile_size=jagged_tile_size),
        out_shape=jax.ShapeDtypeStruct((num_items, m_padded), o_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(1,),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=f"JaggedReduce-tile_{jagged_tile_size}",
    )

    output_padded = kernel(
        *scalar_prefetches,
        pltpu.with_memory_space_constraint(jagged_data_padded, pltpu.HBM),
    )
    return _prepare_outputs(output_padded, m_actual)
