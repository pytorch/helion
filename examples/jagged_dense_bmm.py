"""Jagged-Dense Batch Matrix Multiplication using Helion.

This module implements jagged-dense batch matrix multiplication (BMM)
operation using Helion.

The operation performs batch matrix multiplication between jagged (variable-length)
sequences and dense matrices with optional bias addition. Unlike standard batch
matrix multiplication, the jagged format allows different sequence lengths within
the same batch, making it memory-efficient for variable-length inputs.

Tensor Shapes:
    seq_offsets : [B + 1]  - Cumulative offsets defining sequence boundaries
                             where B is the batch size
    jagged      : [L, D]   - Jagged input tensor where L is the total sum of
                             sequence lengths and D is the embedding dimension
    dense       : [B, D, K] - Dense weight matrices where K is the output dimension
    bias        : [B, K]   - Optional bias vectors
    output      : [L, K]   - Result with same jagged structure as input

Example:
    For a batch of 3 sequences with lengths [2, 1, 3]:
    - seq_offsets = [0, 2, 3, 6]
    - jagged shape = [6, D] (concatenated sequences)
    - dense shape = [3, D, K]
    - output shape = [6, K]

Usage:
    >>> seq_offsets, jagged, dense, bias = random_input(D=32, K=64, batch_size=16)
    >>> output = jagged_dense_bmm(seq_offsets, jagged, dense, bias)
"""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def jagged_dense_bmm(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    L, D = jagged.shape
    B, D, K = dense.shape
    dtype = torch.promote_types(jagged.dtype, dense.dtype)
    device = jagged.device

    jagged = jagged.view(-1)  # flattening to [L * D]
    # Allocate output tensor and flatten to 1D
    output = torch.empty((L, K), dtype=dtype, device=device).view(-1)
    for tile_b in hl.tile(B):
        starts = seq_offsets[tile_b]
        ends = seq_offsets[tile_b.index + 1]
        seq_len = ends - starts
        max_seq_len = seq_len.amax()

        for tile_len in hl.tile(0, max_seq_len):
            mask: torch.Tensor = tile_len.index[None, :] < seq_len[:, None]
            jagged_indices = starts[:, None] + tile_len.index[None, :]

            for tile_k in hl.tile(0, K):
                acc = hl.zeros([tile_b, tile_len, tile_k], dtype=dtype, device=device)
                for tile_d in hl.tile(0, D):
                    jagged_data = hl.load(
                        jagged,
                        [jagged_indices[:, :, None] * D + tile_d.index[None, None, :]],
                        extra_mask=mask[:, :, None] & (tile_d.index[None, None, :] < D),
                    )  # [tile_b, tile_len, tile_d]
                    dense_data = dense[tile_b, tile_d, tile_k]

                    acc = acc + torch.matmul(
                        jagged_data, dense_data
                    )  # [tile_b, tile_len, tile_k]

                if bias is not None:
                    bias_data = bias[tile_b, tile_k]  # [tile_b, tile_k]
                    # [tile_b, tile_len, tile_k] + [tile_b, 1, tile_k] -> [tile_b, tile_len, tile_k]
                    acc = acc + bias_data.unsqueeze(1)

                hl.store(
                    output,
                    [jagged_indices[:, :, None] * K + tile_k.index[None, None, :]],
                    acc,
                    extra_mask=mask[:, :, None],
                )
    return output.reshape(L, K)


def jagged_dense_bmm_reference(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    L, D = jagged.shape
    B, _, K = dense.shape

    # Allocate output tensor
    ref_output = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)

    # Process each example in the batch
    for i in range(B):
        seq_start = seq_offsets[i].item()
        seq_end = seq_offsets[i + 1].item()

        if seq_start < seq_end:  # Non-empty sequence
            seq_data = jagged[seq_start:seq_end]  # [seq_len, D]

            # Matrix multiplication: [seq_len, D] @ [D, K] -> [seq_len, K]
            result = torch.matmul(seq_data, dense[i])

            # Add bias if provided
            if bias is not None:
                result = result + bias[i].unsqueeze(0)

            # Store result
            ref_output[seq_start:seq_end] = result
    return ref_output


def random_input(
    D: int = 4,
    K: int = 5,
    batch_size: int = 3,
    max_seq_len: int = 3,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.randint(
        max_seq_len + 1, size=(batch_size,), device=torch.device("cuda")
    )
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)
    jagged_size = int(seq_offsets[-1].item())
    jagged = (
        torch.empty((jagged_size, D), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-1.0, 1.0)
        .requires_grad_()
    )
    dense = (
        torch.empty((batch_size, D, K), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-1.0, 1.0)
        .requires_grad_()
    )
    bias = (
        torch.empty((batch_size, K), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-1.0, 1.0)
        .requires_grad_()
    )
    return seq_offsets, jagged, dense, bias


def main() -> None:
    seq_offsets, jagged, dense, bias = random_input(
        D=34, K=24, batch_size=23, max_seq_len=37, dtype=torch.float32
    )
    run_example(
        jagged_dense_bmm, jagged_dense_bmm_reference, (seq_offsets, jagged, dense, bias)
    )


if __name__ == "__main__":
    main()
