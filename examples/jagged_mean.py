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
def jagged_mean_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,  # [num_rows] - number of features per row
    max_M: int,  # Maximum number of features
) -> torch.Tensor:
    """
    Compute the mean of each row in a jagged tensor with variable features per row.

    Args:
        x_data: 2-D tensor of shape (total_elements, max_M) holding all elements
        x_offsets: (num_rows + 1) tensor. Row i is the slice
                   x_data[x_offsets[i] : x_offsets[i+1], :]
        x_feature_counts: (num_rows) tensor. Number of valid features for each row
        max_M: Maximum number of features

    Returns:
        2-D tensor of shape (num_rows, max_M) containing the mean of each row.
        Invalid features (beyond x_feature_counts[i]) are set to 0.
    """
    num_rows = x_offsets.size(0) - 1

    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)

    # Flatten x_data for easier indexing
    x_flat = x_data.view(-1)

    # Process rows in tiles
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()

        # Get feature counts for this tile of rows
        feature_counts = x_feature_counts[tile_b]

        # Process features in tiles
        for tile_m in hl.tile(max_M):
            # Create mask for valid features
            feature_valid = tile_m.index < feature_counts[:, None]

            # Initialize accumulator
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            # Process elements within each row
            for tile_k in hl.tile(0, max_nnz):
                # Compute flattened indices
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * max_M + tile_m.index[None, None, :]
                )

                # Combined mask: valid row element AND valid feature
                row_mask = tile_k.index[None, :] < nnz[:, None]
                combined_mask = row_mask[:, :, None] & feature_valid[:, None, :]

                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                # Accumulate - sum across the k dimension (dim=1)
                row_sums = row_sums + x_slice.sum(dim=1)

            # Compute mean
            nnz_float = nnz.to(x_data.dtype)
            nnz_expanded = nnz_float[:, None]

            # Compute result with feature masking
            result = torch.where(nnz_expanded > 0, row_sums / nnz_expanded, 0.0)

            # Apply feature mask to output
            out[tile_b, tile_m] = torch.where(feature_valid, result, 0.0)

    return out


# %%
# Reference Implementation
# ------------------------


# %%
def reference_jagged_mean_kernel_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_M: int,
) -> torch.Tensor:
    """
    PyTorch reference implementation for jagged mean with variable features.

    Args:
        x_data: 2-D tensor holding all elements
        x_offsets: Offsets tensor for row indexing
        x_feature_counts: Number of valid features per row
        max_M: Maximum number of features

    Returns:
        Tensor containing the mean of each row
    """
    num_rows = x_offsets.numel() - 1
    out = torch.zeros((num_rows, max_M), dtype=x_data.dtype, device=x_data.device)
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        num_features = int(x_feature_counts[i])
        if end > start and num_features > 0:
            out[i, :num_features] = x_data[start:end, :num_features].mean(dim=0)
    return out


# %%
# Benchmark Wrapper
# -----------------


# %%
def jagged_mean_tritonbench(
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

    feature_counts = torch.full(
        (B,),
        M,
        dtype=torch.int32,
        device=x_values.device,  # pyright: ignore[reportAttributeAccessIssue]
    )
    return lambda: jagged_mean_kernel(x_values, x_offsets, feature_counts, M)


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
    num_rows, max_cols = 32, 64
    device = DEVICE

    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    nnz = int(x_offsets[-1])
    M = 8  # number of features
    x_data = torch.randn(nnz, M, dtype=torch.float32, device=device)
    feature_counts = torch.randint(
        1, M + 1, (num_rows,), dtype=torch.int32, device=device
    )

    run_example(
        lambda x, o, fc, m: jagged_mean_kernel(x, o, fc, m),
        lambda x, o, fc, m: reference_jagged_mean_kernel_pytorch(x, o, fc, m),
        (x_data, x_offsets, feature_counts, M),
    )


if __name__ == "__main__":
    main()
