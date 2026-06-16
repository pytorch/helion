"""
Jagged Amax Example
===================

Per-row max reduction over a jagged 2-D tensor (variable rows, fixed
feature dim).  Same scaffolding as ``jagged_sum.py``; replaces the sum
accumulator with ``torch.maximum`` and ``-inf`` init.
"""

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def jagged_amax_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """Per-row ``amax`` over a jagged 2-D tensor.

    Args:
        x_data: 2-D tensor of shape ``(total_elements, M)`` holding all rows.
        x_offsets: ``(num_rows + 1)`` cumulative offsets.

    Returns:
        2-D tensor ``(num_rows, M)`` with each row's per-feature max.
    """
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1

    out = torch.full(
        [num_rows, M], float("-inf"), dtype=x_data.dtype, device=x_data.device
    )
    x_flat = x_data.view(-1)

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts

        for tile_m in hl.tile(M):
            row_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)

            for tile_k in hl.jagged_tile(nnz):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_indices])
                row_max = torch.maximum(row_max, x_slice.amax(dim=1))

            out[tile_b, tile_m] = row_max

    return out


def reference_jagged_amax_kernel_pytorch(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for ``jagged_amax_kernel``."""
    num_rows = x_offsets.numel() - 1
    M = x_data.size(1)
    out = torch.full(
        (num_rows, M), float("-inf"), dtype=x_data.dtype, device=x_data.device
    )
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            out[i, :] = x_data[start:end, :].amax(dim=0)
    return out


def jagged_amax_tritonbench(
    tb_op: object, x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> Callable[[], torch.Tensor]:
    """TritonBench operator wrapper.

    Args:
        tb_op: TritonBench operator instance.
        x: Nested tensor in jagged format with shape ``(B, *, M)``.
        B: Batch size.
        M: Number of features.
        seqlen: Maximum sequence length.
        sparsity: Sparsity factor (unused).
    """
    x_values = x._values
    # pyrefly: ignore [missing-attribute]
    x_offsets = x._offsets
    return lambda: jagged_amax_kernel(x_values, x_offsets)


def main() -> None:
    B, M, max_seqlen = 8, 128, 64
    device = DEVICE

    from helion._testing import LONG_INT_TYPE

    seq_lengths = torch.randint(1, max_seqlen + 1, (B,), device=device)
    x_offsets = torch.cat(
        [
            torch.zeros(1, dtype=LONG_INT_TYPE, device=device),
            torch.cumsum(seq_lengths, dim=0).to(LONG_INT_TYPE),
        ]
    )
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=torch.float32, device=device)

    run_example(
        jagged_amax_kernel,
        reference_jagged_amax_kernel_pytorch,
        (x_data, x_offsets),
        kernel_name="helion",
        baseline_name="torch",
    )


if __name__ == "__main__":
    main()
