"""
Grouped GEMM Example
====================

This example demonstrates grouped matrix multiplication (GEMM) where multiple
input matrices ``A_i`` (with potentially different numbers of rows ``M_i``)
are multiplied against a single shared weight matrix ``B``. The results are
concatenated in the original group order.

Key ideas used in this implementation:

- Pack all groups' rows into one contiguous tensor ``A_packed`` with shape
  ``[sum(M_i), K]``. This improves memory locality and simplifies indexing.

- Represent group boundaries with ``group_offsets`` (size ``G+1``), so that
  rows for group ``g`` live in ``A_packed[group_offsets[g]:group_offsets[g+1]]``.

- Use data-dependent tiling over the concatenated row dimension to efficiently
  support jagged group sizes (different ``M_i`` per group) without padding.

Two kernels are provided:

1) ``grouped_gemm_jagged`` - a simple kernel that iterates groups and tiles
   dynamically.

2) ``grouped_gemm_jagged_persistent`` - a persistent kernel with dynamic tile
   assignment for better load balancing across streaming multiprocessors (SMs).
"""

# %%
# Imports and Dependencies
# ------------------------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Grouped GEMM Kernel - Basic Implementation
# ------------------------------------------


# %%
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged(
    A_packed: torch.Tensor,  # [total_M, K], where total_M == sum(M_i)
    B: torch.Tensor,  # [K, N] shared across all groups
    group_offsets: torch.Tensor,  # [G+1], int32/int64, row offsets into A_packed
) -> torch.Tensor:  # [total_M, N] concatenated outputs for all groups
    """
    Perform grouped GEMM on jagged inputs using row offsets.

    Args:
        A_packed: Row-wise concatenation of per-group inputs ``A_i``,
            shape ``[sum(M_i), K]``.
        B: Shared weight matrix, shape ``[K, N]``.
        group_offsets: Row offsets delimiting each group within ``A_packed``,
            shape ``[G+1]``. For group ``g``: rows are
            ``start = group_offsets[g]`` to ``end = group_offsets[g+1]``.

    Returns:
        Output tensor of shape ``[sum(M_i), N]`` equal to
        ``torch.cat([A_i @ B for i in groups], dim=0)``.
    """
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2, "K dimension mismatch between A_packed and B"

    out = torch.empty(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1

    # Process each group independently, tiling over its specific M_g dimension
    for g in hl.grid(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        M_g = end - start
        if M_g != 0:
            # Create 2D tiling pattern over output dimensions (M_g x N) for current group
            for tile_m, tile_n in hl.tile([M_g, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                # K-reduction loop: multiply tiles along K dimension
                for tile_k in hl.tile(K):
                    a_blk = A_packed[start + tile_m.index, tile_k]
                    b_blk = B[tile_k, tile_n]
                    # Perform fused multiply-add with FP32 accumulation for numerical stability
                    acc = torch.addmm(acc, a_blk, b_blk)
                # Convert accumulator to output dtype and store result
                out[start + tile_m.index, tile_n] = acc.to(out.dtype)

    return out


# %%
# Grouped GEMM Kernel - Persistent Implementation
# -----------------------------------------------


# %%
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged_persistent(
    A_packed: torch.Tensor,  # [total_M, K]
    B: torch.Tensor,  # [K, N]
    group_offsets: torch.Tensor,  # [G+1], row offsets into A_packed
) -> torch.Tensor:
    """
    Persistent grouped GEMM with dynamic tile metadata computation.

    This variant computes tile assignments dynamically in the kernel,
    similar to TritonBench's WS variant.

    Args:
        A_packed: Packed A, concatenated by rows across groups, ``[sum(M_i), K]``.
        B: Shared weight matrix, ``[K, N]``.
        group_offsets: Row offsets delimiting each group within ``A_packed``.

    Returns:
        Output tensor of shape ``[sum(M_i), N]``.
    """
    # Set worker count to match GPU streaming multiprocessor count
    device = A_packed.device
    if device.type == "xpu":
        # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we will update it to XeCore number.
        num_workers = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    else:
        num_workers = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count

    # Define tunable block sizes for M, N dimensions (auto-tuned at runtime)
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2

    out = torch.zeros(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1

    for worker_id in hl.grid(num_workers):
        # Persistent thread pattern: each worker processes tiles across all groups
        # using strided/interleaved assignment for load balancing.
        # (i.e. each worker takes every num_workers-th tile. e.g., worker 0 takes tiles 0, N, 2N, ...)
        for g in hl.grid(G):
            group_start = group_offsets[g]
            group_end = group_offsets[g + 1]
            m_size = group_end - group_start

            if m_size > 0:
                # Compute tile grid dimensions for current group
                num_m_tiles = (m_size + BLOCK_M - 1) // BLOCK_M
                # Calculate number of N tiles (shared across all groups)
                num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
                num_group_tiles = num_m_tiles * num_n_tiles

                # Distribute tiles among workers using strided access pattern
                for local_tile in hl.grid(num_group_tiles):
                    tile_in_group = local_tile * num_workers + worker_id
                    if tile_in_group < num_group_tiles:
                        # Convert linear tile index to 2D (M, N) tile coordinates
                        m_tile_idx = tile_in_group % num_m_tiles  # pyright: ignore[reportOperatorIssue]
                        n_tile_idx = tile_in_group // num_m_tiles

                        # Compute global memory indices for current tile
                        base_row = group_start + m_tile_idx * BLOCK_M
                        base_col = n_tile_idx * BLOCK_N  # pyright: ignore[reportOperatorIssue]

                        # Generate row and column index ranges for tile access
                        row_idx = base_row + hl.arange(BLOCK_M)
                        col_idx = base_col + hl.arange(BLOCK_N)

                        # Apply boundary masks to handle partial tiles at edges
                        rows_valid = row_idx < group_end
                        cols_valid = col_idx < N

                        # Initialize FP32 accumulator for numerical precision
                        acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)

                        # Iterate over K dimension in blocks for matrix multiplication
                        for k_tile in hl.tile(K):
                            k_idx = k_tile.index

                            # Load tiles from A_packed and B with boundary checking
                            a_blk = hl.load(
                                A_packed,
                                [row_idx, k_idx],
                                extra_mask=rows_valid[:, None],
                            )
                            b_blk = hl.load(
                                B,
                                [k_idx, col_idx],
                                extra_mask=cols_valid[None, :],
                            )

                            # Perform tile-level matrix multiplication and accumulate
                            acc = torch.addmm(acc, a_blk, b_blk)

                        # Write accumulated result to output with boundary masking
                        valid_2d = rows_valid[:, None] & cols_valid[None, :]
                        hl.store(
                            out,
                            [row_idx, col_idx],
                            acc.to(out.dtype),
                            extra_mask=valid_2d,
                        )

    return out


# %%
# Data Preparation Utilities
# --------------------------


# %%
def _pack_group_inputs(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build ``A_packed``, shared ``B``, and ``group_offsets`` from grouped inputs.

    Expectations:
    - All ``A_i`` share the same ``K`` and dtype/device.
    - All groups share the same ``B`` (as produced by TritonBench inputs).

    Returns ``(A_packed, B_shared, group_offsets)`` where
    ``group_offsets`` has length ``G+1`` with ``group_offsets[0] == 0`` and
    ``group_offsets[g+1] - group_offsets[g] == M_g``.
    """
    assert len(group_A) > 0
    device = group_A[0].device
    dtype = group_A[0].dtype

    # Extract shared weight matrix B (same for all groups in TritonBench)
    B_shared = group_B[0]

    # Compute group offsets and concatenate all A matrices row-wise
    M_sizes = [int(a.size(0)) for a in group_A]
    starts = [0]
    for m in M_sizes:
        starts.append(starts[-1] + m)
    group_offsets = torch.tensor(starts, device=device, dtype=torch.int32)
    A_packed = torch.cat(group_A, dim=0).to(device=device, dtype=dtype).contiguous()
    return A_packed, B_shared, group_offsets


# %%
# TritonBench Integration Wrappers
# --------------------------------


# %%
def grouped_gemm_jagged_tritonbench(
    tb_op: object, group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> Callable[[], torch.Tensor]:
    """Adapter for basic grouped GEMM kernel to work with TritonBench benchmark suite."""

    def inner() -> torch.Tensor:
        A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
        return grouped_gemm_jagged(A_packed, B_shared, group_offsets)

    return inner


def grouped_gemm_jagged_persistent_tritonbench(
    tb_op: object, group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> Callable[[], torch.Tensor]:
    """Adapter for persistent grouped GEMM kernel with dynamic work distribution for TritonBench."""

    def inner() -> torch.Tensor:
        A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
        return grouped_gemm_jagged_persistent(
            A_packed,
            B_shared,
            group_offsets,
        )

    return inner


# %%
# Reference Implementation for Validation
# ---------------------------------------


# %%
def _reference_grouped_gemm(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    B_shared = group_B[0]
    outs = [a @ B_shared for a in group_A]
    return torch.cat(outs, dim=0)


def grouped_gemm_jagged_example(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    """
    Wrapper to run grouped_gemm_jagged with unpacked TritonBench inputs.
    """
    A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
    return grouped_gemm_jagged(A_packed, B_shared, group_offsets)


def grouped_gemm_jagged_persistent_example(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    """
    Wrapper to run grouped_gemm_jagged_persistent with unpacked TritonBench inputs.
    """
    A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
    return grouped_gemm_jagged_persistent(A_packed, B_shared, group_offsets)


# %%
# Test Harness and Validation
# ---------------------------


# %%
def main() -> None:
    torch.manual_seed(0)  # Ensure reproducible test results
    device = DEVICE
    dtype = torch.bfloat16
    G = 4  # Number of groups to test
    K, N = 256, 128  # Shared dimensions: K (reduction), N (output columns)
    # Create test data with varying group sizes (M_i = 64, 128, 192, 256)
    group_A = [
        torch.randn(64 * (i + 1), K, device=device, dtype=dtype).contiguous()
        for i in range(G)
    ]
    # Shared weight matrix B replicated for each group (as per TritonBench convention)
    group_B = [torch.randn(K, N, device=device, dtype=dtype).contiguous()] * G

    print("Testing grouped GEMM kernels...")
    run_example(
        grouped_gemm_jagged_example,
        _reference_grouped_gemm,
        (group_A, group_B),
        rtol=1e-2,
        atol=1e-2,
    )
    print("✓ Non-persistent kernel passed")

    run_example(
        grouped_gemm_jagged_persistent_example,
        _reference_grouped_gemm,
        (group_A, group_B),
        rtol=1e-2,
        atol=1e-2,
    )
    print("✓ Persistent kernel passed")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
