"""
Fused GEMM + Reduce-Scatter
===========================
Uses hl.triton_kernel to call Kraken's symm_mem_sync function,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.

Reduce-scatter is the first phase of a two-shot all-reduce pattern.
Each rank computes its local GEMM, then reduces only its assigned slice
from all ranks. Output shape is (M, N // world_size) per rank.

This is useful when the reduced result will be processed locally before
an all-gather, enabling pipelining of computation with communication.
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
        block_sizes=[64, 64],
        num_warps=8,
        num_stages=3,
    ),
    static_shapes=True,
)
def _gemm_reduce_scatter_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    local_buf: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    output: torch.Tensor,
    world_size_tensor: torch.Tensor,
    BLOCK_SIZE_K: hl.constexpr,
    RANK: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    """
    Fused GEMM + reduce-scatter kernel.

    Each rank:
        1. Computes local GEMM tile: C_local[tile_m, tile_n] = A[tile_m, :] @ B[:, tile_n]
        2. Writes result to symmetric memory buffer
        3. Barrier sync
        4. Reduces this tile from all remote buffers
        5. Writes reduced result to output (full M x N buffer)
        6. Final barrier sync

    Note: Output is full M x N; caller extracts this rank's slice.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    # Specialize world_size to make it a compile-time constant
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote tensors inside the kernel
    buf_tuple = torch.ops.symm_mem.get_remote_tensors(local_buf, GROUP_NAME)

    # Two sync phases per tile
    NUM_SYNC_PHASES: hl.constexpr = 2

    # Combined GEMM + reduce in single loop
    for tile_m, tile_n in hl.tile([M, N]):
        # Step 1: Compute local GEMM tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        local_buf[tile_m, tile_n] = acc

        # Step 2: Sync - ensure all ranks have written this tile
        sync_id_base = (tile_m.id * helion.cdiv(N, 64) + tile_n.id) * NUM_SYNC_PHASES
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id_base + 0, RANK, world_size, False, True),
            output_like=None,
        )

        # Step 3: Reduce this tile from all ranks
        reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for buf in buf_tuple:
            reduced = reduced + buf[tile_m, tile_n].to(torch.float32)

        # Step 4: Write full reduced result to output
        output[tile_m, tile_n] = reduced

        # Step 5: Final barrier sync
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id_base + 1, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_gemm_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size_k: int = 64,
) -> torch.Tensor:
    """
    Runs the fused GEMM + reduce-scatter kernel.

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

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K = a.shape
    _, N = b.shape

    if N % world_size != 0:
        raise ValueError(f"N ({N}) must be divisible by world_size ({world_size})")

    # Allocate symmetric memory buffer for full GEMM result
    local_buf = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)

    # Get signal pad for synchronization
    symm_mem_hdl = symm_mem.rendezvous(local_buf, group.group_name)

    # Output buffer is full M x N (kernel writes full result, we extract slice)
    output = torch.empty((M, N), dtype=torch.float32, device=a.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=a.device)

    _gemm_reduce_scatter_kernel(
        a,
        b,
        local_buf,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        world_size_tensor,
        BLOCK_SIZE_K=block_size_k,
        RANK=symm_mem_hdl.rank,
        GROUP_NAME=group.group_name,
    )

    # Extract this rank's slice for reduce-scatter semantics
    size_per_rank = N // world_size
    my_slice_start = rank * size_per_rank
    my_slice_end = my_slice_start + size_per_rank

    return output[:, my_slice_start:my_slice_end].contiguous()


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

    print(f"[Rank {rank}] Running Helion GEMM + reduce-scatter...")
    helion_result = helion_gemm_reduce_scatter(a, b)

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

    # N must be divisible by world_size (4)
    test(M=512, N=256, K=128, device=device)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/gemm_reduce_scatter_fused.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
