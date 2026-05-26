"""Pallas/TPU template for jagged reductions over the second-minor axis.

Computes::

    out[i, :] = x_data[x_offsets[i] : x_offsets[i + 1], :].sum(axis=0)

I/O layout (HBM)::

    x_data    : [L, M_padded]  values, M_padded multiple of 128 (lane-aligned)
    x_offsets : i32[B + 1]     cumulative offsets, x_offsets[B] == L
    out       : [B, M_padded]  one row per item

Lowering shape:
  - Grid: ``(num_b,)`` — one program per item.
  - ``x_offsets`` is scalar-prefetched into SMEM. The SPU reads
    ``offsets[b], offsets[b+1]`` per program and passes a dynamic
    ``pl.ds(start, len)`` into the input BlockSpec's index_map.
  - Input BlockSpec uses ``pl.BoundedSlice(k_sz)`` for the seq dim,
    so the per-program DMA moves only the live rows (rounded up to
    8-sublane alignment); the trailing VMEM region is undefined.
  - Mosaic handles the double-buffered DMA automatically; no manual
    ``pltpu.make_async_copy`` / semaphore orchestration.

In-kernel work:
  - Re-read ``offsets[b], offsets[b+1]`` from SMEM to recover the actual
    live row count (``actual``).
  - Mask the partial-tail rows ``iota < actual`` (0 is the sum identity).
  - Reduce + cast + store to the single output row.

This template is intentionally narrow (only ``sum`` over a lane-padded M);
it will be extended with a ``reduction_kind`` parameter once a second
reduction (mean / layer_norm) needs the same scaffolding.
"""
from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


_LANE_ALIGN = 128
_SUBLANE_ALIGN = 8


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


@functools.partial(
    jax.jit,
    static_argnames=("k_sz", "o_dtype", "vmem_limit_bytes"),
)
def jagged_sum_pallas(
    x_data: jax.Array,       # [L, M_padded]  values
    x_offsets: jax.Array,    # i32[B + 1]     cumulative offsets
    *,
    k_sz: int = 64,          # BoundedSlice upper bound (rows per program)
    o_dtype: Any = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:              # [B, M_padded]
    """Jagged sum over the second-minor axis. See module docstring."""
    if o_dtype is None:
        o_dtype = x_data.dtype
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    L, M = x_data.shape
    num_b = x_offsets.shape[0] - 1

    if M % _LANE_ALIGN != 0:
        raise ValueError(
            f"M={M} must be a multiple of {_LANE_ALIGN} (TPU lane alignment); "
            f"caller is responsible for zero-padding the M dim before calling "
            f"the kernel."
        )
    if k_sz % _SUBLANE_ALIGN != 0:
        raise ValueError(
            f"k_sz={k_sz} must be a multiple of {_SUBLANE_ALIGN} (TPU sublane "
            f"alignment for BoundedSlice)."
        )

    # ------------------------------------------------------------------ #
    # Specs.                                                             #
    #                                                                    #
    # PrefetchScalarGridSpec convention (cf. JAX tests/pallas/            #
    # tpu_pallas_test.py:test_trivial_scalar_prefetch):                   #
    #   - index_map signature: (grid_indices..., scalar_refs...)         #
    #   - kernel body signature: (scalar_refs..., in_refs..., out_refs,  #
    #                             scratch_refs...)                       #
    #                                                                    #
    # Input: dynamic per-program slab via BlockSpec+BoundedSlice+ds.     #
    # Output: kept in HBM (memory_space=pltpu.HBM). Mosaic enforces a     #
    # block-shape alignment rule — last two dims divisible by 8 and 128, #
    # or equal to the array dims — and a (1, M) per-program output block #
    # violates the sublane rule. Same shape gmm.py uses at the top level: #
    # output stays in HBM, the kernel does an explicit per-program DMA   #
    # writeback via pltpu.make_async_copy from a VMEM staging buffer.    #
    # ------------------------------------------------------------------ #
    def x_index_map(b_id, x_offsets_ref):
        # SPU scalar reads from SMEM, fed into pl.ds for the DMA.
        start = x_offsets_ref[b_id]
        end = x_offsets_ref[b_id + 1]
        return (pl.ds(start, end - start), 0)

    x_spec = pl.BlockSpec(
        (pl.BoundedSlice(k_sz), M),
        x_index_map,
    )
    out_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    scratch_shapes = [
        pltpu.VMEM((1, M), o_dtype),     # output staging row
        pltpu.SemaphoreType.DMA,         # out-write semaphore
    ]

    # --------------------------------------------------- #
    # Inner kernel: masked reduce + explicit DMA out.     #
    # --------------------------------------------------- #
    def kernel(x_offsets_ref, x_ref, out_ref, out_buf, out_sem):
        b_id = pl.program_id(0)
        start = x_offsets_ref[b_id]
        end = x_offsets_ref[b_id + 1]
        actual = end - start

        # The BoundedSlice block has static shape [k_sz, M]; only the first
        # `actual` rows are live (Mosaic rounds DMA up to a sublane boundary,
        # so rows [actual, align_to(actual, 8)) may hold live garbage from a
        # prior DMA; rows [align_to(actual, 8), k_sz) are uninitialized).
        # Masking by `iota < actual` zero-fills both regions before sum.
        iota = lax.broadcasted_iota(jnp.int32, (k_sz, 1), 0)
        mask = iota < actual
        x = jnp.where(mask, x_ref[...], jnp.zeros_like(x_ref[...]))
        partial = x.sum(axis=0, dtype=jnp.float32).astype(o_dtype)
        out_buf[0, :] = partial

        cp = pltpu.make_async_copy(
            out_buf,
            out_ref.at[pl.ds(b_id, 1), :],
            out_sem,
        )
        cp.start()
        cp.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_b, M), o_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[x_spec],
            out_specs=out_spec,
            scratch_shapes=scratch_shapes,
            grid=(num_b,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=f"JaggedSum-k_{k_sz}",
    )(x_offsets, x_data)
