"""
Fused GEMM + True Reduce-Scatter (WIP - Does Not Work Yet)
===========================================================
Uses hl.triton_kernel to call Kraken's symm_mem_sync function,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.

STATUS: This implementation does NOT work due to Helion limitations.
The issue is that we cannot conditionally skip output writes for tiles
that don't belong to this rank's slice. All blocks in the tile loop
execute and write to their computed output indices, causing out-of-bounds
writes for tiles outside the rank's slice.

WHAT WOULD BE NEEDED TO FIX THIS:
1. Helion support for conditional stores (tl.where or masking at the block level)
2. OR: A way to launch only a subset of blocks (not all tiles in the grid)
3. OR: Separate kernels for GEMM and reduce phases

For a working reduce-scatter implementation, see gemm_reduce_scatter_fused.py
which does a full all-reduce and extracts the slice at the end.

Reduce-scatter is the first phase of a two-shot all-reduce pattern.
Each rank computes its local GEMM, then reduces only its assigned slice
from all ranks. Output shape is (M, N // world_size) per rank.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion._testing import DEVICE
import helion.language as hl

# Add kraken to path to import symm_mem_sync
sys.path.insert(0, "/data/users/willfeng/kraken")
from kraken._ptx_utils.symm_mem_barrier import symm_mem_sync  # noqa: E402


@helion.jit(
    config=helion.Config(
        block_sizes=[64, 64],  # [M tiles, N tiles]
        num_warps=8,
        num_stages=3,
    ),
    static_shapes=True,
)
def _gemm_reduce_scatter_true_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    local_buf: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    output: torch.Tensor,
    world_size_tensor: torch.Tensor,
    BLOCK_SIZE_K: hl.constexpr,
    SIZE_PER_RANK: hl.constexpr,
    RANK: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    """
    Fused GEMM + true reduce-scatter kernel.

    This kernel uses a single loop over the full (M, N) space.
    - Phase 1: GEMM + sync for all tiles
    - Phase 2: Reduce + write only for tiles in this rank's slice

    The output write is guarded by checking if the tile is in this rank's slice.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    # Specialize world_size to make it a compile-time constant
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote tensors inside the kernel
    buf_tuple = torch.ops.symm_mem.get_remote_tensors(local_buf, GROUP_NAME)

    # This rank's slice boundaries in the full N dimension
    my_slice_start = RANK * SIZE_PER_RANK
    my_slice_end = my_slice_start + SIZE_PER_RANK

    # Number of N tiles per rank (for determining which tiles belong to this rank)
    # Assuming block_size_n = 64 and SIZE_PER_RANK is a multiple of 64
    BLOCK_SIZE_N: hl.constexpr = 64
    num_tiles_per_rank = SIZE_PER_RANK // BLOCK_SIZE_N
    my_first_tile_n = RANK * num_tiles_per_rank
    my_last_tile_n = my_first_tile_n + num_tiles_per_rank

    # Single loop over all tiles
    for tile_m, tile_n in hl.tile([M, N]):
        # Step 1: Compute local GEMM tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        local_buf[tile_m, tile_n] = acc

        # Step 2: Sync - ensure all ranks have written this tile
        sync_id = tile_m.id * helion.cdiv(N, BLOCK_SIZE_N) + tile_n.id
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id, RANK, world_size, False, True),
            output_like=None,
        )

        # Step 3: Reduce this tile from all ranks
        reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for buf in buf_tuple:
            reduced = reduced + buf[tile_m, tile_n].to(torch.float32)

        # Step 4: Write to output ONLY if this tile is in this rank's slice
        # This rank owns tiles with tile_n.id in [my_first_tile_n, my_last_tile_n)
        # Output column = tile_n.id - my_first_tile_n (in tile units)
        #
        # We use a "masked" store by writing to output with offset calculation
        # If the tile is outside our slice, we write to a dummy location (clamped)
        #
        # For tiles in our slice:
        #   output_tile_id = tile_n.id - my_first_tile_n  (0, 1, 2, ...)
        #   output_col_start = output_tile_id * BLOCK_SIZE_N
        #
        # For tiles outside our slice:
        #   We still compute output_tile_id but it will be negative or >= num_tiles_per_rank
        #   The store will be out of bounds - Triton should mask this, but may not
        #
        # Alternative: duplicate the store with different offsets
        # For now, we just write to the computed position - this may cause issues
        # if Helion/Triton doesn't properly mask out-of-bounds stores
        output_tile_id = tile_n.id - my_first_tile_n
        output[tile_m, output_tile_id * BLOCK_SIZE_N : (output_tile_id + 1) * BLOCK_SIZE_N] = reduced

        # Step 5: Final sync
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_gemm_reduce_scatter_true(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size_k: int = 64,
) -> torch.Tensor:
    """
    Runs the fused GEMM + true reduce-scatter kernel.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        block_size_k: Block size for K dimension tiling

    Returns:
        Reduced scatter result of shape (M, N // world_size)
        Each rank gets a different slice of the reduced GEMM result.
    """
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("Example currently supports float32 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")

    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    world_size = dist.get_world_size()
    M, K = a.shape
    _, N = b.shape

    if N % world_size != 0:
        raise ValueError(f"N ({N}) must be divisible by world_size ({world_size})")

    size_per_rank = N // world_size

    # Allocate symmetric memory buffer for full GEMM result
    local_buf = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)

    # Get signal pad for synchronization
    symm_mem_hdl = symm_mem.rendezvous(local_buf, group.group_name)

    # Output is only this rank's slice: M x size_per_rank
    output = torch.empty((M, size_per_rank), dtype=torch.float32, device=a.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=a.device)

    return _gemm_reduce_scatter_true_kernel(
        a,
        b,
        local_buf,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        world_size_tensor,
        BLOCK_SIZE_K=block_size_k,
        SIZE_PER_RANK=size_per_rank,
        RANK=symm_mem_hdl.rank,
        GROUP_NAME=group.group_name,
    )


def reference_gemm_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation using separate PyTorch operations.

    Returns the local rank's slice of the reduced result.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each rank computes full GEMM
    local_gemm = torch.matmul(a, b)

    # All-reduce to get sum across all ranks
    dist.all_reduce(local_gemm)

    # Extract this rank's slice (reduce-scatter result)
    N = local_gemm.shape[1]
    size_per_rank = N // world_size
    my_slice_start = rank * size_per_rank
    my_slice_end = my_slice_start + size_per_rank

    return local_gemm[:, my_slice_start:my_slice_end].contiguous()


def test(M: int, N: int, K: int, device: torch.device) -> None:
    """Test the Helion implementation against the reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert N % world_size == 0, f"N ({N}) must be divisible by world_size ({world_size})"

    # Use same seed across ranks for consistent B matrix
    torch.manual_seed(42)
    b = torch.randn((K, N), dtype=torch.float32, device=device)

    # Different A per rank (simulating distributed input)
    torch.manual_seed(42 + rank)
    a = torch.randn((M, K), dtype=torch.float32, device=device)

    print(f"[Rank {rank}] Running Helion true GEMM + reduce-scatter...")
    helion_result = helion_gemm_reduce_scatter_true(a, b)

    # Sync before reference
    torch.cuda.synchronize()
    dist.barrier()

    print(f"[Rank {rank}] Running reference...")
    reference = reference_gemm_reduce_scatter(a, b)

    print(f"[Rank {rank}] Comparing results...")
    print(f"[Rank {rank}] Helion output shape: {helion_result.shape}")
    print(f"[Rank {rank}] Reference output shape: {reference.shape}")

    torch.testing.assert_close(helion_result, reference, rtol=1e-1, atol=1e-1)
    print(f"[Rank {rank}] Results match!")


def main() -> None:
    # Only NVSHMEM backend implements get_remote_tensors for now
    symm_mem.set_backend("NVSHMEM")

    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    # Enable symmetric memory for the group
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # N must be divisible by world_size (4) and size_per_rank must be divisible by block_size_n (64)
    # N=256, world_size=4 => size_per_rank=64, which is divisible by 64
    test(M=512, N=256, K=128, device=device)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/gemm_reduce_scatter_true.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
