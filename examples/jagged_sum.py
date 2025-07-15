from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def jagged_sum_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    max_M_tensor: torch.Tensor,  # Dummy tensor whose size indicates max features
) -> torch.Tensor:
    """
    Compute the sum of each row in a jagged tensor.

    Args
    ----
    x_data          : 2-D tensor of shape (total_elements, max_M) holding all elements.
    x_offsets       : (num_rows + 1) tensor. Row i is the slice
                      x_data[x_offsets[i] : x_offsets[i+1], :].
    max_M_tensor    : Dummy tensor whose numel() gives max number of features.

    Returns
    -------
    result : 2-D tensor of shape (num_rows, max_M) containing the sum of each row.
    """
    num_rows = x_offsets.size(0) - 1
    max_M = max_M_tensor.numel()  # Extract max features from dummy tensor

    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)

    # Flatten x_data for easier indexing
    x_flat = x_data.view(-1)

    # Process rows in tiles
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()

        # Process features in tiles
        for tile_m in hl.tile(max_M):
            # Initialize accumulator
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)

            # Process elements within each row
            for tile_k in hl.tile(0, max_nnz):
                # Compute flattened indices
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * max_M + tile_m.index[None, None, :]
                )

                # Mask for valid row elements
                row_mask = tile_k.index[None, :] < nnz[:, None]
                combined_mask = row_mask[:, :, None]

                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                # Accumulate - sum across the k dimension (dim=1)
                row_sums = row_sums + x_slice.sum(dim=1)

            # Store result
            out[tile_b, tile_m] = row_sums

    return out


def reference_jagged_sum_kernel_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    max_M: int,
) -> torch.Tensor:
    """PyTorch reference implementation for jagged sum."""
    num_rows = x_offsets.numel() - 1
    out = torch.zeros((num_rows, max_M), dtype=x_data.dtype, device=x_data.device)
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            out[i, :] = x_data[start:end, :].sum(dim=0)
    return out


def jagged_sum_tritonbench(
    x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> dict[str, torch.Tensor]:
    """
    Wrapper for tritonbench that matches the expected interface.

    Args:
        x: Nested tensor in jagged format with shape (B, *, M)
        B: Batch size
        M: Number of features
        seqlen: Maximum sequence length
        sparsity: Sparsity factor (not used)

    Returns:
        Dictionary with "output" key containing tensor of shape (B, M) with sum values per row
    """
    x_values = x._values
    x_offsets = x._offsets  # pyright: ignore[reportAttributeAccessIssue]
    
    max_M_tensor = torch.empty(M, device=x_values.device)  # pyright: ignore[reportAttributeAccessIssue]

    output = jagged_sum_kernel(x_values, x_offsets, max_M_tensor)
    return {"output": output}


def main() -> None:
    num_rows, max_cols = 32, 64
    device = "cuda"

    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    nnz = int(x_offsets[-1])
    M = 8  # number of features
    x_data = torch.randn(nnz, M, dtype=torch.float32, device=device)
    max_M_tensor = torch.empty(M, device=device)

    run_example(
        lambda x, o, mt: jagged_sum_kernel(x, o, mt),
        lambda x, o, mt: reference_jagged_sum_kernel_pytorch(x, o, mt.numel()),
        (x_data, x_offsets, max_M_tensor),
    )


if __name__ == "__main__":
    main()