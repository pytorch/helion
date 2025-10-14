"""
Jagged Layer Normalization Example
==================================

This example demonstrates how to compute layer normalization on jagged tensors
using Helion. The implementation closely follows the torch_jagged_layer_norm_torch_sum
algorithm from tritonbench but is optimized for Helion's tiling approach.

A jagged tensor is a nested tensor where each sequence can have different lengths.
Layer normalization is applied across the feature dimension (last dimension) for
each individual sequence, computing mean and variance only over valid elements.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import itertools
from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Jagged Layer Norm Kernel
# ------------------------


# %%
@helion.kernel(autotune_effort="none")
def jagged_layer_norm_kernel(
    x_values: torch.Tensor,  # [total_L, M] - compressed values
    x_offsets: torch.Tensor,  # [B+1] - sequence start offsets
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute layer normalization on jagged tensor using Helion.

    This kernel implements layer normalization for jagged tensors by:
    1. Computing mean and variance for each sequence individually
    2. Normalizing values within each sequence
    3. Applying optional affine transformation (weight/bias)

    Args:
        x_values: Compressed values tensor of shape [total_L, M]
        x_offsets: Sequence boundary offsets of shape [B+1]
        eps: Small value for numerical stability

    Returns:
        Normalized tensor of same shape as x_values [total_L, M]
    """
    total_L, M = x_values.shape
    B = x_offsets.size(0) - 1

    # Output tensor
    out = torch.empty_like(x_values)

    x_flat = x_values.view(-1)
    out_flat = out.view(-1)

    # Process sequences in tiles
    for tile_b in hl.tile(B):
        # Get sequence boundaries for this tile
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seq_lengths = ends - starts
        max_seq_len = seq_lengths.amax()

        # Initialize accumulators for mean and variance computation
        mean_acc = hl.zeros([tile_b], dtype=x_values.dtype)
        var_acc = hl.zeros([tile_b], dtype=x_values.dtype)

        # First pass: compute mean
        for tile_m in hl.tile(M):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.tile(0, max_seq_len):
                # Compute indices into x_values
                indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = indices[:, :, None] * M + tile_m.index[None, None, :]

                # Create mask for valid elements
                row_mask = tile_k.index[None, :] < seq_lengths[:, None]
                combined_mask = row_mask[:, :, None]

                # Load values with masking
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )

                # Accumulate sum for mean (sum across sequence dimension)
                row_sums = row_sums + x_slice.sum(dim=1)
            mean_acc = mean_acc + row_sums.sum(dim=1)
        seq_lengths_float = seq_lengths.to(x_values.dtype)
        mean_acc = mean_acc / (seq_lengths_float * M)

        # Second pass: compute variance
        for tile_m in hl.tile(M):
            var_sums = hl.zeros([tile_b, tile_m], dtype=x_values.dtype)
            for tile_k in hl.tile(0, max_seq_len):
                # Compute indices into x_values
                indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = indices[:, :, None] * M + tile_m.index[None, None, :]

                # Create mask for valid elements
                row_mask = tile_k.index[None, :] < seq_lengths[:, None]
                combined_mask = row_mask[:, :, None]

                # Load values with masking
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )

                # Compute centered values
                centered = torch.where(
                    combined_mask,
                    x_slice.to(torch.float32) - mean_acc[:, None, None],
                    0.0,
                )

                # Accumulate squared differences for variance
                var_sums = var_sums + (centered * centered).sum(dim=1)
            var_acc = var_acc + var_sums.sum(dim=1)

        # Compute variance and reciprocal standard deviation
        variance = var_acc / (seq_lengths_float * M)
        rstd = torch.rsqrt(variance + eps)

        # Third pass: compute layernorm
        for tile_m in hl.tile(M):
            for tile_k in hl.tile(0, max_seq_len):
                # Compute indices into x_values
                indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = indices[:, :, None] * M + tile_m.index[None, None, :]

                # Create mask for valid elements
                row_mask = tile_k.index[None, :] < seq_lengths[:, None]
                combined_mask = row_mask[:, :, None]

                # Load values with masking
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )

                # Normalize
                normalized = torch.where(
                    combined_mask,
                    (x_slice.to(torch.float32) - mean_acc[:, None, None])
                    * rstd[:, None, None],
                    0.0,
                )

                # Store result
                hl.store(
                    out_flat,
                    [flat_indices],
                    normalized.to(x_values.dtype),
                    extra_mask=combined_mask,
                )

    return out.reshape(total_L, M)


# %%
# Reference Implementation
# ------------------------


# %%
def reference_jagged_layer_norm_pytorch(
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Simple reference implementation using unbind approach for validation.
    """

    return torch.cat(
        [
            torch.nn.functional.layer_norm(
                x_values[x_offsets[i] : x_offsets[i + 1], :],
                x_values[x_offsets[i] : x_offsets[i + 1], :].shape,  # pyright: ignore[reportArgumentType]
                eps=eps,
            )
            for i in range(x_offsets.shape[0] - 1)
        ],
        dim=0,
    )


# %%
# Benchmark Wrapper
# -----------------


# %%
def jagged_layer_norm_tritonbench(
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
        Callable that returns normalized tensor values
    """
    x_values = x._values
    x_offsets = x._offsets  # pyright: ignore[reportAttributeAccessIssue]

    return lambda: jagged_layer_norm_kernel(x_values, x_offsets, eps=1e-6)


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
    Main entry point for jagged layer norm example.

    Creates test data and compares the Helion implementation against
    both PyTorch reference implementations.
    """
    # B, M, max_seqlen = 3, 4, 3
    B_list = [2**n for n in list(range(5, 16, 3))]
    M_list = [2**n for n in list(range(5, 10, 3))]
    max_seqlen_list = [128]
    eps = 1e-6
    device = DEVICE

    for B, M, max_seqlen in itertools.product(B_list, M_list, max_seqlen_list):
        x_data, x_offsets = create_test_jagged_tensor(
            B, M, max_seqlen, device, dtype=torch.float32
        )
        run_example(
            lambda x, o, eps: jagged_layer_norm_kernel(x, o, eps),
            lambda x, o, eps: reference_jagged_layer_norm_pytorch(x, o, eps),
            (x_data, x_offsets, eps),
        )


# %%
if __name__ == "__main__":
    main()
