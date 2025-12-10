"""
Two-Shot All-Reduce + Bias + RMS Norm Fusion
=============================================
Uses hl.triton_kernel to call Kraken's symm_mem_sync function,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.

Two-shot pattern is more scalable for larger messages (150KB-15MB).

Maps to: kraken/fused/two_shot_all_reduce_bias_rms_norm.py
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
        block_sizes=[8],  # Process multiple rows at a time (fewer sync IDs)
        num_warps=8,
    ),
    static_shapes=True,
)
def two_shot_allreduce_bias_rmsnorm_kernel(
    x: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    output: torch.Tensor,
    world_size_tensor: torch.Tensor,
    EPS: hl.constexpr,
    RANK: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    """
    Fused two-shot all-reduce + bias addition + RMS normalization.

    Two-shot pattern:
        1. Barrier sync
        2. Reduce-scatter: each rank reduces its slice [rank*size_per_rank : (rank+1)*size_per_rank]
        3. All-gather: write reduced slice to all buffers
        4. Barrier sync
        5. RMS Norm on full row
        6. Final barrier sync
    """
    N, D = x.size()
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote buffers from all ranks
    buf_tuple = torch.ops.symm_mem.get_remote_tensors(x, GROUP_NAME)

    # Size per rank for reduce-scatter
    size_per_rank = D // world_size

    # We have 3 sync phases per tile, so we need different sync IDs to avoid races
    # sync_id = tile_n.id * 3 + phase (0, 1, or 2)
    NUM_SYNC_PHASES: hl.constexpr = 3

    for tile_n in hl.tile(N):
        # Compute base sync_id for this tile (each tile uses 3 consecutive sync IDs)
        base_sync_id = tile_n.id * NUM_SYNC_PHASES

        # Phase 1: Barrier - ensure all ranks have written their data
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, base_sync_id + 0, RANK, world_size, False, True),
            output_like=None,
        )

        # Phase 2: Reduce-scatter
        # Each rank handles its slice: [RANK * size_per_rank : (RANK + 1) * size_per_rank]
        my_slice_start = RANK * size_per_rank
        my_slice_end = my_slice_start + size_per_rank

        # Start with bias for this slice (use [None, :] to broadcast to 2D)
        # Initialize acc with zeros matching 2D shape, then add bias
        acc = x[tile_n, my_slice_start:my_slice_end].to(torch.float32) * 0.0 + bias[None, my_slice_start:my_slice_end].to(torch.float32)

        # Sum contributions from all ranks for this slice
        for buf in buf_tuple:
            acc = acc + buf[tile_n, my_slice_start:my_slice_end].to(torch.float32)

        # Phase 3: All-gather - write reduced slice back to all buffers
        for buf in buf_tuple:
            buf[tile_n, my_slice_start:my_slice_end] = acc.to(x.dtype)

        # Phase 4: Barrier - ensure all ranks have written their reduced slices
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, base_sync_id + 1, RANK, world_size, True, True),
            output_like=None,
        )

        # Phase 5: RMS Normalization
        # Read the full row from local buffer (now contains all-reduced data)
        row = x[tile_n, :].to(torch.float32)

        # Per-row variance (reduce over D dimension)
        variance = torch.mean(row * row, dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + EPS)
        normalized = row * rstd
        output[tile_n, :] = (normalized * weight[None, :].to(torch.float32)).to(x.dtype)

        # Phase 6: Final barrier
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, base_sync_id + 2, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_two_shot_allreduce_bias_rmsnorm(
    x_shared: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Wrapper that sets up symmetric memory and calls the Helion kernel.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    symm_mem_hdl = symm_mem.rendezvous(x_shared, group.group_name)

    # Verify D is divisible by world_size for reduce-scatter
    D = x_shared.shape[-1]
    world_size = symm_mem_hdl.world_size
    assert D % world_size == 0, (
        f"Hidden dim {D} must be divisible by world_size {world_size}"
    )

    N, D = x_shared.shape
    output = torch.empty((N, D), dtype=x_shared.dtype, device=x_shared.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=x_shared.device)

    return two_shot_allreduce_bias_rmsnorm_kernel(
        x_shared,
        bias,
        weight,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        world_size_tensor,
        EPS=eps,
        RANK=symm_mem_hdl.rank,
        GROUP_NAME=group.group_name,
    )


def reference_two_shot_allreduce_bias_rmsnorm(
    x: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Reference implementation using separate PyTorch operations.
    """
    x_reduced = x.clone()
    dist.all_reduce(x_reduced)
    x_with_bias = x_reduced + bias

    # RMS Norm
    variance = x_with_bias.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normalized = x_with_bias.to(torch.float32) * rstd
    return (normalized * weight.to(torch.float32)).to(x.dtype)


def test(N: int, D: int, device: torch.device, dtype: torch.dtype) -> None:
    """Test the Helion implementation against the reference."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert D % world_size == 0, (
        f"Hidden dim {D} must be divisible by world_size {world_size}"
    )

    # Create symmetric memory tensor
    x_shared = symm_mem.empty(N, D, dtype=dtype, device=device).normal_()

    # Bias and weight are the same across ranks (inference use case)
    torch.manual_seed(42)
    bias = torch.randn(D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)

    # Reference: use a regular tensor (not symmetric memory) for NCCL all_reduce
    x_ref = torch.empty(N, D, dtype=dtype, device=device)
    x_ref.copy_(x_shared)

    print(f"[Rank {rank}] Running Helion two-shot allreduce + bias + rmsnorm...")
    result_helion = helion_two_shot_allreduce_bias_rmsnorm(x_shared, bias, weight)

    # Sync before reference to ensure Helion kernel is complete on all ranks
    torch.cuda.synchronize()
    dist.barrier()

    print(f"[Rank {rank}] Running reference...")
    result_ref = reference_two_shot_allreduce_bias_rmsnorm(x_ref, bias, weight)

    print(f"[Rank {rank}] Comparing results...")
    torch.testing.assert_close(result_helion, result_ref, rtol=1e-4, atol=1e-4)
    print(f"[Rank {rank}] Results match!")


def main() -> None:
    symm_mem.set_backend("NVSHMEM")
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

    # D must be divisible by world_size (4)
    test(N=128, D=4096, device=device, dtype=torch.float32)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/two_shot_allreduce_bias_rmsnorm.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
