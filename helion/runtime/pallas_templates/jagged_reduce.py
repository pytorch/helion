"""Pallas/TPU template for jagged_sum.

L-space grid: programs iterate over fixed-size ``(block_L, block_M)``
slabs of the input. Inside each program, an inner ``lax.fori_loop`` walks
all ``N`` items, clipping each to the current L-block via SMEM-resident
``x_offsets``, then mask-reduces and accumulates into a single shared
output block.

Key choice: the output BlockSpec is ``(N, block_M)`` with ``index_map ->
(0, j)`` — i.e. **all L-programs share the same output block** (per
M-block). Pallas's read-modify-write through HBM makes the per-program
``out_ref[item_idx, :] += partial`` accumulate correctly across the
L-programs that touch each item. Block shape equals the array's first
dim (N), which satisfies Mosaic's sublane-alignment rule via the
"equal to array dim" escape clause — for any N.

This avoids three things we tried unsuccessfully:
  - ``pl.BoundedSlice`` (not supported at top-level ``pl.pallas_call``)
  - ``pltpu.emit_pipeline`` + sublane bitcast (not needed for static
    blocks; only needed if we wanted dynamic per-program slabs)
  - manual ``pltpu.make_async_copy`` orchestration (BlockSpec + the
    "equal to array dim" escape clause handle alignment naturally).

Computes::

    out[i, :] = x_data[x_offsets[i] : x_offsets[i + 1], :].sum(axis=0)

I/O layout (HBM)::

    x_data    : [L, M]       values (M must divide block_M)
    x_offsets : i32[N + 1]   cumulative, x_offsets[N] == L
    out       : [N, M]       per-item sums

``L`` is internally zero-padded up to a multiple of ``block_L``; the
padding rows are never referenced by any item (since x_offsets values
are bounded by L), so they don't affect any sum.

Cost model:
  - Per L-program: ``N`` inner-loop iterations, of which only the few
    items touching this L-block do real work (the rest hit
    ``pl.when(lo < hi)`` and skip). Inner-loop cost ~ O(N).
  - Output block size in VMEM: ``N * block_M * sizeof(dtype)``. For N
    large enough that this exceeds VMEM, switch to a different grid
    layout. For typical jagged workloads (B in the hundreds), fits
    easily.

Adapted from the L-space panel in jagged_sum_loop_structures.html
(committed earlier in this branch).
"""
from __future__ import annotations

import functools
from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _jagged_sum_pallas_kernel(
    offsets_ref: jax.Array,  # SMEM ref, int32[N + 1]
    x_ref: jax.Array,        # VMEM slab, [block_L, block_M]
    out_ref: jax.Array,      # VMEM block, [N, block_M]
    *,
    N: int,
    block_L: int,
):
    """Inner Pallas kernel. See module docstring for the design."""
    i_L = pl.program_id(0)
    l_start = i_L * block_L

    # Zero-init the shared output block once per (j) M-program. All
    # L-programs in this M-column share the same out_ref via the
    # index_map -> (0, j) and accumulate into it. Without this, the
    # first read of HBM is undefined.
    @pl.when(i_L == 0)
    def _init_out():
        out_ref[...] = jnp.zeros_like(out_ref[...])

    def loop_body(item_idx, _):
        it_start = offsets_ref[item_idx]
        it_end = offsets_ref[item_idx + 1]

        # Overlap between item [it_start, it_end) and this L-block.
        lo = jnp.maximum(it_start, l_start) - l_start
        hi = jnp.minimum(it_end, l_start + block_L) - l_start

        @pl.when(lo < hi)
        def _accumulate():
            iota = lax.iota(jnp.int32, block_L)
            mask = jnp.logical_and(iota >= lo, iota < hi)
            # Arithmetic mask, not jnp.where: boolean broadcast trips a
            # Mosaic layout bug for some block shapes.
            mask_f = mask.astype(x_ref.dtype)
            partial = (mask_f[:, None] * x_ref[...]).sum(axis=0)
            out_ref[item_idx, :] += partial

    lax.fori_loop(0, N, loop_body, None)


@functools.partial(
    jax.jit,
    static_argnames=("block_L", "block_M", "o_dtype", "vmem_limit_bytes"),
)
def jagged_sum_pallas(
    x_data: jax.Array,       # [L, M]
    x_offsets: jax.Array,    # i32[N + 1]
    *,
    block_L: int = 64,
    block_M: int = 128,
    o_dtype: Any = None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:              # [N, M]
    """JAX-native wrapper around the jagged-sum Pallas kernel.

    See module docstring for the kernel design. This wrapper handles
    L-padding and assembles the pallas_call.
    """
    if o_dtype is None:
        o_dtype = x_data.dtype

    L_raw, M = x_data.shape
    N = x_offsets.shape[0] - 1

    if M % block_M != 0:
        raise ValueError(
            f"M={M} must be a multiple of block_M={block_M}"
        )

    # Zero-pad x_data so total rows is a multiple of block_L. Padding rows
    # are never referenced (no item spans into them) so they don't affect
    # correctness; they just round the input out to a static-block grid.
    L_pad = ((L_raw + block_L - 1) // block_L) * block_L
    if L_pad != L_raw:
        x_data = jnp.pad(x_data, ((0, L_pad - L_raw), (0, 0)))

    grid = (L_pad // block_L, M // block_M)

    compiler_params_kwargs: dict[str, Any] = {"disable_bounds_checks": True}
    if vmem_limit_bytes is not None:
        compiler_params_kwargs["vmem_limit_bytes"] = vmem_limit_bytes

    return pl.pallas_call(
        functools.partial(
            _jagged_sum_pallas_kernel,
            N=N,
            block_L=block_L,
        ),
        out_shape=jax.ShapeDtypeStruct((N, M), o_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=grid,
            in_specs=[
                pl.BlockSpec((block_L, block_M), lambda i, j, _: (i, j)),
            ],
            out_specs=pl.BlockSpec(
                (N, block_M), lambda i, j, _: (0, j)
            ),
            scratch_shapes=[],
        ),
        compiler_params=pltpu.CompilerParams(**compiler_params_kwargs),
        name=f"JaggedSum-L{block_L}-M{block_M}",
    )(x_offsets, x_data)
