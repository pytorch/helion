"""
Jagged Mean Example
===================

This example demonstrates how to compute the mean of each row in a jagged tensor
with variable features per row using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Jagged Mean Kernel
# ------------------


# %%
@helion.kernel()
def jagged_sum_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean of each row in a jagged tensor with variable features per row.

    Args:
        x_data: 2-D tensor of shape (total_elements, M) holding all elements
        x_offsets: (num_rows + 1) tensor. Row i is the slice
                   x_data[x_offsets[i] : x_offsets[i+1], :]

    Returns:
        2-D tensor of shape (num_rows, M) containing the sum of jagged dimension.
    """
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1

    out = torch.zeros([num_rows, M], dtype=x_data.dtype, device=x_data.device)

    # Flatten x_data for easier indexing
    x_flat = x_data.view(-1)

    # Process rows in tiles
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()

        # Process features in tiles
        for tile_m in hl.tile(M):
            # Initialize accumulator
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            # Process elements within each row
            for tile_k in hl.tile(0, max_nnz):
                # Compute flattened indices
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )

                # Combined mask: valid row element AND valid feature
                row_mask = tile_k.index[None, :] < nnz[:, None]
                combined_mask = row_mask[:, :, None]

                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                # Accumulate - sum across the k dimension (dim=1)
                row_sums = row_sums + x_slice.sum(dim=1)

            # Apply feature mask to output
            out[tile_b, tile_m] = row_sums

    return out


# %%
# Reference Implementation
# ------------------------


# %%
def reference_jagged_sum_kernel_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference implementation for jagged mean with variable features.

    Args:
        x_data: 2-D tensor holding all elements
        x_offsets: Offsets tensor for row indexing

    Returns:
        Tensor containing the mean of each row
    """
    num_rows = x_offsets.numel() - 1
    M = x_data.size(1)
    out = torch.zeros((num_rows, M), dtype=x_data.dtype, device=x_data.device)
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            out[i, :] = x_data[start:end, :].sum(dim=0)
    return out


# %%
# Benchmark Wrapper
# -----------------


# %%
def jagged_sum_tritonbench(
    tb_op: object, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches the expected interface.

    Args:
        tb_op: TritonBench operator instance
        x: Nested tensor in jagged format with shape (B, *, M)
        B: Batch size
        M: Number of features
        seqlen: Maximum sequence length
        sparsity: Sparsity factor (not used)

    Returns:
        Callable that returns tensor of shape (B, M) with mean values per row and feature
    """
    x_values = x._values
    x_offsets = x._offsets  # pyright: ignore[reportAttributeAccessIssue]

    return lambda: jagged_sum_kernel(x_values, x_offsets)


# %%
# Helper function to create test data
# -----------------------------------


# %%
def create_test_jagged_tensor(
    B: int,
    M: int,
    max_seqlen: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create test jagged tensor data."""

    # Generate random sequence lengths
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,), device=device)

    # Create offsets
    x_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(seq_lengths, dim=0),
        ]
    )

    # Create values
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=dtype, device=device)

    return x_data, x_offsets


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the jagged mean kernel verification.

    Creates test data with random jagged tensors and feature counts, then compares
    the kernel implementation against the PyTorch reference implementation.
    """
    B, M, max_seqlen = 8, 128, 64
    device = DEVICE

    x_data, x_offsets = create_test_jagged_tensor(
        B, M, max_seqlen, device, dtype=torch.float32
    )

    run_example(
        lambda x, o: jagged_sum_kernel(x, o),
        lambda x, o: reference_jagged_sum_kernel_pytorch(x, o),
        (x_data, x_offsets),
    )


if __name__ == "__main__":
    main()
