"""Pallas/TPU template for jagged_sum, derived from the RPA v3 design.

Single-program grid: one Pallas program drives the entire computation
internally via ``pl.loop`` over sequences. No cross-program state, no
shared output blocks, no reliance on Pallas's BlockSpec residency
across programs — so the accumulator pattern has no race surface.

Sums elements across the jagged axis::

    out[i, :] = x_data[x_offsets[i] : x_offsets[i + 1], :].sum(axis=0)

I/O (HBM)::

    x_data    : x.dtype[L, M_actual]   values
    x_offsets : i32[B + 1]              cumulative offsets, x_offsets[B] == L
    out       : x.dtype[B, M_actual]    per-sequence sums

Internally::

    1. ``prepare_inputs`` lane-pads M_actual up to M_padded (multiple of 128).
       Zero is the sum identity, so padded lanes contribute nothing.
    2. The kernel runs as a single program (grid=(1,)). Inside, ``pl.loop``
       iterates over b in [0, num_b). Each iteration:
         - prefetches the NEXT b's first k-block (double-buffered)
         - waits for THIS b's current k-block
         - reduces it into the fp32 VMEM accumulator (mask the partial tail)
         - on the last k-block of this b: flushes the accumulator out
           through a separate double-buffered VMEM->HBM DMA
    3. ``prepare_outputs`` trims back to M_actual.

Building blocks (N3-N8 from the original RPA v3 distillation):
  - N3: ``pltpu.make_async_copy`` wrapper
  - N4: degenerate-copy semaphore wait
  - N5: re-derive (b_start, b_end, b_len) inside closures from b_idx
  - N6: ``bo_ids_ref[2]`` records the b_idx each output buffer is holding
  - N7: ``sem_ids_ref[2]`` ping-pong rotation in SMEM
  - N8: ``@pl.when`` guards on acc init (k_idx == 0) and bo flush
        (k_idx == num_k - 1), i.e. at the SEQ boundary, not per k-block

Known limitation: an empty sequence (x_offsets[i+1] == x_offsets[i]) leaves
the corresponding output row stale because num_k == 0 skips the inner
loop entirely — no init, no flush. Callers should ensure no empty
sequences, or pre-zero the output buffer.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_LANE_ALIGN = 128


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _cdiv(x, y):
    return (x + y - 1) // y


class _IterQuantities(NamedTuple):
    """Per-sequence bounds in x_data L-space."""

    b_start: jax.Array
    b_end: jax.Array
    b_len: jax.Array


def _derive_iter_quantities(b_idx, x_offsets_ref) -> _IterQuantities:
    b_start = x_offsets_ref[b_idx]
    b_end = x_offsets_ref[b_idx + 1]
    b_len = b_end - b_start
    return _IterQuantities(b_start=b_start, b_end=b_end, b_len=b_len)


def _jagged_sum_kernel(*args, **kwargs):
    """Outer dispatcher: iterate b_idx in [0, num_b) via pl.loop."""
    x_offsets_ref = args[0]
    num_b = x_offsets_ref.shape[0] - 1

    @pl.loop(0, num_b)
    def _(b_idx):
        return _jagged_sum_kernel_loop(b_idx, *args, **kwargs)


def _jagged_sum_kernel_loop(
    b_idx,
    # Prefetch (SMEM)
    x_offsets_ref,   # i32[max_num_b + 1]
    sem_ids_ref,     # i32[2] — (k_sem_idx, bo_sem_idx); ping-pong rotation (N7)
    bo_ids_ref,      # i32[2] — b_idx per bo buffer (N6)
    # Input (HBM)
    x_hbm_ref,       # x.dtype[L, m_padded]
    # Output (HBM)
    o_hbm_ref,       # o_dtype[num_b, m_padded]
    # Scratch (VMEM)
    bk_x2_ref,       # x.dtype[2, k_sz, m_padded]  — input double-buffer
    bo_x2_ref,       # o_dtype[2, m_padded]        — output double-buffer
    sems,            # DMA semaphores[2, 2]        — (k | bo) × 2 ping-pong
    acc_ref,         # fp32[m_padded]              — per-seq accumulator
    *,
    k_sz,            # tile_k block size (rows of x per DMA)
):
    o_dtype = bo_x2_ref.dtype
    _, m_padded = x_hbm_ref.shape
    num_b = x_offsets_ref.shape[0] - 1

    (b_start, b_end, b_len) = _derive_iter_quantities(b_idx, x_offsets_ref)

    # -------------------------------------------------------------------- #
    # DMA wrappers (N3 verbatim + N4 degenerate-copy wait)                  #
    # -------------------------------------------------------------------- #

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_k(b_idx, k_idx, k_sem_idx, *, wait=False):
        """HBM x_data → VMEM bk_x2 ping-pong slot."""
        sem = sems.at[0, k_sem_idx]
        vmem_ref = bk_x2_ref.at[k_sem_idx]

        qs = _derive_iter_quantities(b_idx, x_offsets_ref)
        x_row_start = qs.b_start + k_idx * k_sz
        sz = jnp.minimum(k_sz, qs.b_end - x_row_start)

        if not wait:
            _async_copy(
                x_hbm_ref.at[pl.ds(x_row_start, sz), :],
                vmem_ref.at[pl.ds(0, sz), :],
                sem,
                wait=False,
            )
        else:
            # Degenerate copy as semaphore wait (N4)
            dst = vmem_ref.at[pl.ds(0, sz), :]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _send_bo(b_idx, bo_sem_idx, *, wait=False):
        """VMEM bo_x2 slot → HBM out[b_idx, :]."""
        sem = sems.at[1, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]

        if not wait:
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[b_idx, :],
                sem,
                wait=False,
            )
        else:
            # Degenerate copy as semaphore wait (N4)
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def start_fetch_k(b_idx, k_idx, k_sem_idx):
        return _fetch_k(b_idx, k_idx, k_sem_idx)

    def wait_fetch_k(b_idx, k_idx, k_sem_idx):
        return _fetch_k(b_idx, k_idx, k_sem_idx, wait=True)

    def start_send_bo(b_idx, bo_sem_idx):
        # Record what this buffer is currently holding (N6) — wait_send_bo
        # will read it back to know what to wait on.
        bo_ids_ref[bo_sem_idx] = b_idx
        _send_bo(b_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_b_idx = bo_ids_ref[bo_sem_idx]
        # Guard: skip at cold start (-1 sentinel) and skip if the prior
        # buffer is already past us.
        @pl.when(jnp.logical_and(0 <= old_b_idx, old_b_idx <= b_idx))
        def _():
            _send_bo(old_b_idx, bo_sem_idx, wait=True)

    # -------------------------------------------------------------------- #
    # Compute body: reduce a tile_k block and accumulate                    #
    # -------------------------------------------------------------------- #

    def process():
        num_k = _cdiv(b_len, k_sz)

        def get_next_k_ids(b_idx, k_idx, k_sem_idx):
            """Compute (next_b, next_k, next_sem) for prefetch look-ahead."""
            next_k_idx = k_idx + 1
            is_last_k = next_k_idx == num_k
            next_k_idx = lax.select(is_last_k, 0, next_k_idx)
            next_b_idx = lax.select(is_last_k, b_idx + 1, b_idx)
            next_k_sem_idx = lax.select(k_sem_idx == 0, 1, 0)
            return next_b_idx, next_k_idx, next_k_sem_idx

        @pl.loop(0, num_k, unroll=False)
        def compute_with_k(k_idx):
            # N8: acc init at seq boundary (not per-k-block)
            @pl.when(k_idx == 0)
            def init_acc():
                acc_ref[...] = jnp.zeros_like(acc_ref)

            k_sem_idx = sem_ids_ref[0]
            next_b_idx, next_k_idx, next_k_sem_idx = get_next_k_ids(
                b_idx, k_idx, k_sem_idx
            )

            # Prefetch next k-block (N7: swap SMEM index, then start fetch)
            @pl.when(next_b_idx < num_b)
            def prefetch_next_k():
                sem_ids_ref[0] = next_k_sem_idx
                start_fetch_k(next_b_idx, next_k_idx, next_k_sem_idx)

            # Wait for cur k-block
            wait_fetch_k(b_idx, k_idx, k_sem_idx)

            # Compute slot: load → masked reduce → accumulate (N1, plain
            # add — no exp/rescale because we're a sum, not a softmax)
            k_row_start = k_idx * k_sz
            sz = jnp.minimum(k_sz, b_len - k_row_start)
            k_block = bk_x2_ref[k_sem_idx]  # (k_sz, m_padded)
            # Mask off the partial-tail rows: identity for sum is 0
            iota = lax.broadcasted_iota(jnp.int32, (k_sz, 1), 0)
            mask = iota < sz
            k_masked = jnp.where(mask, k_block, jnp.zeros_like(k_block))
            acc_ref[...] = acc_ref[...] + k_masked.sum(
                axis=0, dtype=jnp.float32
            )

            # N8: bo flush at seq boundary (not per-k-block)
            @pl.when(k_idx == num_k - 1)
            def flush_bo():
                # Wait for the OTHER bo buffer's prior send to drain so we
                # can reuse it. Then advance ping-pong index in SMEM (N7).
                bo_sem_idx = sem_ids_ref[1]
                sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
                wait_send_bo(bo_sem_idx)

                # Cast acc → o_dtype, store to bo buffer, kick off DMA
                bo_x2_ref.at[bo_sem_idx][...] = acc_ref[...].astype(o_dtype)
                start_send_bo(b_idx, bo_sem_idx)

    @pl.when(b_idx == 0)
    def prologue():
        # Cold-start the prefetch chain (the @pl.loop-driven prefetch_next_k
        # has no "previous" iteration to fire for b_idx=0).
        start_fetch_k(b_idx=0, k_idx=0, k_sem_idx=0)

    @pl.when(b_idx < num_b)
    def pipeline():
        process()

    @pl.when(b_idx == num_b - 1)
    def epilogue():
        # Drain both bo ping-pong buffers before the kernel returns,
        # otherwise the last seq's outputs may not be flushed to HBM.
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)


def _prepare_inputs(x_data, m_actual):
    m_padded = _align_to(m_actual, _LANE_ALIGN)
    if m_padded != m_actual:
        x_data = jnp.pad(
            x_data,
            ((0, 0), (0, m_padded - m_actual)),
            constant_values=0,
        )
    return x_data


def _prepare_outputs(o_padded, m_actual):
    return o_padded[:, :m_actual]


@functools.partial(
    jax.jit,
    static_argnames=("o_dtype", "k_sz", "vmem_limit_bytes"),
)
def jagged_sum_pallas(
    x_data: jax.Array,           # [L, M_actual]
    x_offsets: jax.Array,        # i32[B + 1]
    *,
    o_dtype: Any = None,
    k_sz: int = 64,              # rows of x per DMA block
    vmem_limit_bytes: int | None = None,
) -> jax.Array:                  # [B, M_actual]
    """Jagged sum over the second-minor axis. See module docstring."""
    if o_dtype is None:
        o_dtype = x_data.dtype
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    _, m_actual = x_data.shape
    num_b = x_offsets.shape[0] - 1

    # Lane-pad the input M dim (zero-pad is the sum identity).
    x = _prepare_inputs(x_data, m_actual)
    _, m_padded = x.shape

    # Initial SMEM state — N7 ping-pong indices start at 0; N6 buffer
    # ownership initialized to -1 (sentinel: "buffer never used").
    init_sem_ids = jnp.zeros((2,), jnp.int32)
    init_bo_ids = jnp.full((2,), -1, jnp.int32)

    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)

    # Scratch: input double-buffer, output double-buffer, semaphores,
    # fp32 accumulator.
    bk_double_buf = pltpu.VMEM((2, k_sz, m_padded), x.dtype)
    bo_double_buf = pltpu.VMEM((2, m_padded), o_dtype)
    acc_scratch = pltpu.VMEM((m_padded,), jnp.float32)
    scratch_shapes = [
        bk_double_buf,                       # bk_x2_ref
        bo_double_buf,                       # bo_x2_ref
        pltpu.SemaphoreType.DMA((2, 2)),     # sems: 2 fetch types × 2 buffers
        acc_scratch,                         # acc_ref
    ]

    scalar_prefetches = (x_offsets, init_sem_ids, init_bo_ids)

    kernel = pl.pallas_call(
        functools.partial(_jagged_sum_kernel, k_sz=k_sz),
        out_shape=jax.ShapeDtypeStruct((num_b, m_padded), o_dtype),
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
        name=f"JaggedSum-k_{k_sz}",
    )

    o_padded = kernel(
        *scalar_prefetches,
        pltpu.with_memory_space_constraint(x, pltpu.HBM),
    )
    return _prepare_outputs(o_padded, m_actual)
