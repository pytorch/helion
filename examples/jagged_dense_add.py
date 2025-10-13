"""
Jagged Dense Addition Example
=============================

This example demonstrates how to implement an addition operation between a jagged tensor
and a dense tensor using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Jagged Tensor Format
# --------------------

# %%
"""
A tensor x is stored in a jagged-row, prefix-sparse layout that packs only the non-zero
elements of each row. All non-zeros are concatenated into a one-dimensional buffer
x_data, ordered row-by-row. A companion index array x_offsets of length num_rows + 1
provides random access: for any row i, the slice x_data[x_offsets[i] : x_offsets[i+1]]
contains exactly the first K_i non-zero entries of that row (with K_i = x_offsets[i+1]
− x_offsets[i]). Elements beyond column K_i − 1 are implicitly zero and therefore
omitted from storage.
"""


# %%
# Jagged Dense Addition Kernel
# ----------------------------


# %%
@helion.kernel()
def jagged_dense_add_2d(
    x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Add a jagged-prefix sparse tensor (x_data, x_offsets) to a dense matrix y
    and return the dense result.

    Args:
        x_data: 1-D tensor holding all non-zero elements row-by-row
        x_offsets: (num_rows + 1) tensor. Row i is the slice
                   x_data[x_offsets[i] : x_offsets[i+1]] (length K_i)
        y: (num_rows, N) tensor, N >= max(K_i)

    Returns:
        Dense tensor of shape (num_rows, N) containing the sum of the jagged and dense tensors
    """
    num_rows = y.size(0)
    assert x_offsets.size(0) == num_rows + 1
    out = torch.zeros_like(y)
    for tile0 in hl.tile(num_rows):
        starts = x_offsets[tile0]
        ends = x_offsets[tile0.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        # Note, the dynamic loop bounds aren't strictly necessary for this example, since
        # the output is dense, and we iterate over the rest in the next loop. However,
        # it is useful to illustrate how more complex jagged+jagged ops can be handled.
        for tile1 in hl.tile(0, max_nnz):
            x_slice = hl.load(
                x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None],
            )
            out[tile0, tile1] = y[tile0, tile1] + x_slice
        for tile1 in hl.tile(max_nnz, out.size(1)):
            # fill in any leftover columns with y
            out[tile0, tile1] = y[tile0, tile1]
    return out


# %%
# Reference Implementation
# ------------------------


# %%
def jagged_dense_add_2d_reference(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of jagged dense addition in pure PyTorch.

    Args:
        x_data: 1-D tensor holding all non-zero elements row-by-row
        x_offsets: (num_rows + 1) tensor with offsets for each row
        y: Dense tensor to add to the jagged tensor

    Returns:
        Dense tensor containing the sum of the jagged and dense tensors
    """
    num_rows = x_offsets.numel() - 1
    assert y.shape[0] == num_rows
    out = y.clone()
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        out[i, 0 : end - start] += x_data[start:end]
    return out


# %%
# Utility Function
# ----------------


# %%
def random_jagged_2d(
    num_rows: int,
    max_cols: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random jagged 2D tensor data.

    Args:
        num_rows: Number of rows in the jagged tensor
        max_cols: Maximum number of columns per row
        dtype: Data type for the tensor values
        device: Device to create the tensors on

    Returns:
        Tuple of (x_data, x_offsets) where:
            - x_data: 1-D tensor holding all non-zeros row-by-row
            - x_offsets: (num_rows+1) tensor with offsets for each row
    """
    # random positive K_i for each row
    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    # prefix-sum -> offsets
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    # total nnz
    nnz = int(x_offsets[-1])
    # random non-zero data
    x_data = torch.randn(nnz, dtype=dtype, device=device)
    return x_data, x_offsets


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the jagged dense add kernel verification.

    Creates random jagged 2D data and a dense tensor, then compares the kernel
    implementation against the PyTorch reference implementation.
    """
    rows, cols = 256, 5000
    x_data, x_offsets = random_jagged_2d(rows, cols, device=DEVICE)
    y = torch.randn([rows, cols], device=DEVICE)

    run_example(
        jagged_dense_add_2d, jagged_dense_add_2d_reference, (x_data, x_offsets, y)
    )


if __name__ == "__main__":
    main()
