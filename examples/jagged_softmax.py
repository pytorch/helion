"""
Jagged Softmax Example
======================

This example demonstrates how to compute the softmax across each batch in a jagged tensor using Helion.
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
# Reference Implementation
# ------------------------


# %%
def reference_jagged_softmax_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference implementation for jagged softmax.

    Args:
        x_data: 2-D tensor holding all elements
        x_offsets: Offsets tensor for row indexing

    Returns:
        Tensor containing the per-batch softmax scores (same shape as x_data)
    """
    vals = []
    for i, j in itertools.pairwise(x_offsets):
        y = x_data[i:j]
        vals.append(torch.softmax(y, dim=0))
    return torch.cat(vals, dim=0)


# %%
# Jagged Softmax Kernel
# ---------------------


# %%
@helion.kernel()
def jagged_softmax_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-batch softmax in a jagged tensor.

    Args:
        x_data: 2-D tensor of shape (total_elements, max_M) holding all elements
        x_offsets: (num_rows + 1) tensor. Row i is the slice
                   x_data[x_offsets[i] : x_offsets[i+1], :]

    Returns:
        2-D tensor of shape (total_elements, max_M), containing the per-batch softmax scores.
    """
    N = int(x_offsets[-1].item())
    num_rows, M = x_offsets.size(0) - 1, x_data.size(1)
    out = torch.zeros(N * M, dtype=x_data.dtype, device=x_data.device)

    # flatten
    x_flat = x_data.view(-1)

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seqlens = ends - starts
        max_seqlen = seqlens.amax()

        for tile_m in hl.tile(M):
            block_max = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)
            block_new_max = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)
            block_L = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)

            for tile_k in hl.tile(max_seqlen):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )
                row_mask = tile_k.index[None, :] < seqlens[:, None]
                combined_mask = row_mask[:, :, None] & (tile_m.index < M)[None, None, :]
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                slice_max = torch.where(combined_mask, x_slice, float("-inf")).amax(
                    dim=1
                )
                block_new_max = torch.maximum(block_max, slice_max)
                block_L *= torch.exp(block_max - block_new_max)
                block_L += torch.exp(
                    torch.where(
                        combined_mask,
                        x_slice - block_new_max[:, None, :],
                        float("-inf"),
                    )
                ).sum(dim=1)
                block_max = block_new_max

            for tile_k in hl.tile(max_seqlen):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )
                row_mask = tile_k.index[None, :] < seqlens[:, None]
                combined_mask = row_mask[:, :, None] & (tile_m.index < M)[None, None, :]
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                block_out = (
                    torch.exp(x_slice - block_max[:, None, :]) / block_L[:, None, :]
                )
                hl.store(
                    out,
                    [flat_indices],
                    block_out,
                    extra_mask=combined_mask,
                )

    return out.reshape(N, M)


# %%
# Benchmark Wrapper
# -----------------


# %%
def jagged_softmax_tritonbench(
    tb_op: object, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches the expected interface.

    Args:
        tb_op: TritonBench operator instance
        x: Nested tensor in jagged format with shape (B, *, M)
        B: Batch size (unused)
        M: Number of features (unused)
        seqlen: Maximum sequence length (unused)
        sparsity: Sparsity factor (unused)

    Returns:
        Callable that returns tensor of shape (N, M), where N = total number of rows in the jagged tensor
    """
    return lambda: jagged_softmax_kernel(x._values, x._offsets)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point for jagged softmax kernel verification.
    """
    num_rows, max_cols = 512, 64
    device = DEVICE

    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    nnz = int(x_offsets[-1])
    M = 128  # number of features
    x_data = torch.randn(nnz, M, dtype=torch.float32, device=device)

    out_eager = reference_jagged_softmax_pytorch(x_data, x_offsets)
    out_hl = jagged_softmax_kernel(x_data, x_offsets)
    assert torch.allclose(out_eager, out_hl)

    run_example(
        lambda x, o: jagged_softmax_kernel(x, o),
        lambda x, o: reference_jagged_softmax_pytorch(x, o),
        (x_data, x_offsets),
    )


if __name__ == "__main__":
    main()
