"""
Jagged Sum Example (TPU optimized Helion DSL)
=============================================

This example computes the sum of each row in a jagged tensor using a purely
static L-space grid, written entirely in the Helion DSL. 
The DMA fetches perfectly aligned dense blocks from the L-dimension, and the 
kernel resolves item boundaries within VMEM dynamically using masking.
"""

from __future__ import annotations
from typing import Callable
import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def jagged_sum_kernel_tpu(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    TPU-optimized jagged sum kernel using the Helion DSL.

    Args:
        x_data: 2-D tensor of shape (L, M) holding all elements
        x_offsets: (N + 1) tensor. Row i is the slice
                   x_data[x_offsets[i] : x_offsets[i+1], :]

    Returns:
        2-D tensor of shape (N, M) containing the sum of jagged dimension.
    """
    M = x_data.shape[1]
    N = x_offsets.shape[0] - 1
    L = x_data.shape[0]

    out = torch.zeros([N, M], dtype=x_data.dtype, device=x_data.device)

    # Static L-space grid: Avoids dynamic loop bounds on the MXUs
    for tile_l in hl.tile(L):
        l_start = tile_l.index
        block_L = tile_l.block_size

        for tile_m in hl.tile(M):
            # DMA fetches perfectly aligned block_L x block_M chunk from HBM
            chunk = x_data[tile_l, tile_m]

            # Resolve item boundaries dynamically within VMEM
            for item_idx in range(N):
                it_start = x_offsets[item_idx]
                it_end = x_offsets[item_idx + 1]

                # We can determine validity purely by checking if the global row index 
                # falls within this batch item's start and end offsets!
                global_row = l_start + hl.arange(block_L)
                mask = (global_row >= it_start) & (global_row < it_end)

                # Arithmetic masking bypasses boolean broadcast layout constraints.
                # Multiplying by 1.0 implicitly casts the boolean mask to a float 
                # safely within the Helion AST without using .to()
                safe_chunk = (mask * 1.0)[:, None] * chunk
                partial = safe_chunk.sum(dim=0)

                # Accumulate the partial sum for this batch item
                out[item_idx, tile_m] += partial

    return out


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
    
    # TPUs only support up to 32-bit integers
    x_offsets = x_offsets.to(torch.int32)
    
    return x_data, x_offsets


def main() -> None:
    B, M, max_seqlen = 8, 128, 64
    device = DEVICE

    x_data, x_offsets = create_test_jagged_tensor(
        B, M, max_seqlen, device, dtype=torch.float32
    )

    # Pad x_data to a safe large multiple (e.g. 1024) to avoid TPU hardware memory faults 
    # when the Helion compiler's dynamic block_L size (e.g. 128, 256, 512) overhangs the 
    # end of the flat array during its static tile loads.
    nnz = x_data.shape[0]
    MAX_BLOCK_SIZE = 1024
    padded_L = ((nnz + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE
    
    x_padded = torch.zeros((padded_L, M), dtype=x_data.dtype, device=x_data.device)
    x_padded[:nnz, :] = x_data

    run_example(
        lambda x, o: jagged_sum_kernel_tpu(x, o),
        lambda x, o: reference_jagged_sum_kernel_pytorch(x, o),
        (x_padded, x_offsets),
    )


if __name__ == "__main__":
    main()