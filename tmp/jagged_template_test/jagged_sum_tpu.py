"""
Jagged Sum Example (TPU optimized)
==================================

This example computes the sum of each row in a jagged tensor using a purely
static L-space grid, as detailed in the `jagged_sum_loop_structures.html` document.
The DMA fetches perfectly aligned dense blocks, and the kernel resolves item 
boundaries within VMEM dynamically.
"""

from __future__ import annotations
from typing import Callable
import functools

import torch
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import helion
from helion._testing import DEVICE
from helion._testing import run_example


def jagged_sum_pallas_kernel(
    offsets_ref: jax.Array,  # [N + 1] (Prefetched to SMEM)
    x_ref: jax.Array,        # [block_L, block_M] (VMEM slab)
    out_ref: jax.Array,      # [N, block_M]
    *,
    N: int,
    block_L: int,
):
    i_L = pl.program_id(0)
    l_start = i_L * block_L
    
    # Walk all items, accumulate into out_ref for each one this block touches.
    # We use lax.fori_loop to keep the unroll size manageable for large N.
    def loop_body(item_idx, _):
        # Because offsets_ref is in SMEM (via PrefetchScalarGridSpec),
        # this scalar read is perfectly valid.
        it_start = offsets_ref[item_idx]
        it_end   = offsets_ref[item_idx + 1]
        
        # Calculate overlap between this item and the current L-block
        lo = jnp.maximum(it_start, l_start) - l_start
        hi = jnp.minimum(it_end, l_start + block_L) - l_start

        # If this item has rows in this L-block
        @pl.when(lo < hi)
        def _accumulate():
            valid_len = hi - lo
            iota = lax.iota(jnp.int32, block_L)
            mask = jnp.logical_and(iota >= lo, iota < hi)
            
            # Arithmetic masking bypasses the boolean broadcast layout crash in Mosaic
            mask_float = mask.astype(x_ref.dtype)
            safe_x = mask_float[:, None] * x_ref[...]
            partial = safe_x.sum(axis=0)
            
            out_ref[item_idx, :] += partial

    lax.fori_loop(0, N, loop_body, None)


def jagged_sum_kernel_tpu(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    block_L: int = 64,
    block_M: int = 128,
) -> torch.Tensor:
    
    M = x_data.shape[1]
    N = x_offsets.shape[0] - 1
    
    # Pad x_data to ensure it divides perfectly by block_L
    nnz = x_data.shape[0]
    padded_L = ((nnz + block_L - 1) // block_L) * block_L
    
    x_padded = torch.zeros((padded_L, M), dtype=x_data.dtype, device=x_data.device)
    x_padded[:nnz, :] = x_data
    
    x_jax = jnp.asarray(x_padded.cpu().numpy())
    offsets_jax = jnp.asarray(x_offsets.cpu().numpy())
    
    grid = (padded_L // block_L, M // block_M)
    
    # We use PrefetchScalarGridSpec to force offsets_ref into SMEM
    kernel = pl.pallas_call(
        functools.partial(
            jagged_sum_pallas_kernel,
            N=N,
            block_L=block_L,
        ),
        out_shape=jax.ShapeDtypeStruct((N, M), x_jax.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=grid,
            in_specs=[
                pl.BlockSpec(
                    (block_L, block_M),
                    lambda i, j, _: (i, j)
                )
            ],
            out_specs=pl.BlockSpec(
                (N, block_M),
                lambda i, j, _: (0, j)
            ),
            scratch_shapes=[],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
    )
    
    out_jax = kernel(offsets_jax, x_jax)
    return torch.from_numpy(jax.device_get(out_jax)).to(x_data.device)


# %%
# Reference Implementation
# ------------------------

def reference_jagged_sum_kernel_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    num_rows = x_offsets.numel() - 1
    M = x_data.size(1)
    out = torch.zeros((num_rows, M), dtype=x_data.dtype, device=x_data.device)
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            out[i, :] = x_data[start:end, :].sum(dim=0)
    return out


def create_test_jagged_tensor(
    B: int,
    M: int,
    max_seqlen: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,), device=device)
    x_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(seq_lengths, dim=0),
    ])
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=dtype, device=device)
    return x_data, x_offsets


def main() -> None:
    B, M, max_seqlen = 8, 128, 64
    device = DEVICE

    x_data, x_offsets = create_test_jagged_tensor(
        B, M, max_seqlen, device, dtype=torch.float32
    )

    run_example(
        lambda x, o: jagged_sum_kernel_tpu(x, o),
        lambda x, o: reference_jagged_sum_kernel_pytorch(x, o),
        (x_data, x_offsets),
    )


if __name__ == "__main__":
    main()
