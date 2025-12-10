"""
One-Shot All-Reduce + Bias + RMS Norm Fusion
=============================================
Uses hl.triton_kernel to call Kraken's symm_mem_sync function,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.

Maps to: kraken/fused/one_shot_all_reduce_bias_rms_norm.py
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
def one_shot_allreduce_bias_rmsnorm_kernel(
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
    Fused one-shot all-reduce + bias addition + RMS normalization.

    Performs:
        1. Barrier sync (ensure x is visible to all ranks)
        2. All-reduce: acc = bias + sum(x from all ranks)
        3. RMS Norm: y = acc * rsqrt(mean(acc^2) + eps) * weight
        4. Final barrier sync
    """
    N, D = x.size()
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote tensors from all ranks
    x_tuple = torch.ops.symm_mem.get_remote_tensors(x, GROUP_NAME)

    for tile_n in hl.tile(N):
        # Sync: ensure all ranks have written their data
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, tile_n.id, RANK, world_size, False, True),
            output_like=None,
        )

        # All-reduce + bias: acc = bias + sum(x from all ranks)
        # x[tile_n, :] is 2D [tile_size, D], bias is 1D [D]
        acc = x[tile_n, :].to(torch.float32) * 0.0 + bias[None, :].to(torch.float32)
        for remote_x in x_tuple:
            acc = acc + remote_x[tile_n, :].to(torch.float32)

        # RMS Norm: y = acc * rsqrt(mean(acc^2) + eps) * weight
        # Per-row variance (reduce over D dimension)
        variance = torch.mean(acc * acc, dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + EPS)
        normalized = acc * rstd
        output[tile_n, :] = (normalized * weight[None, :].to(torch.float32)).to(x.dtype)

        # Final sync
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, tile_n.id, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_one_shot_allreduce_bias_rmsnorm(
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

    N, D = x_shared.shape
    output = torch.empty((N, D), dtype=x_shared.dtype, device=x_shared.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=x_shared.device)

    return one_shot_allreduce_bias_rmsnorm_kernel(
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


def reference_one_shot_allreduce_bias_rmsnorm(
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
    rank = dist.get_rank()

    # Create symmetric memory tensor
    x_shared = symm_mem.empty(N, D, dtype=dtype, device=device).normal_()

    # Bias and weight are the same across ranks (inference use case)
    torch.manual_seed(42)
    bias = torch.randn(D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)

    # Reference: use a regular tensor (not symmetric memory) for NCCL all_reduce
    x_ref = torch.empty(N, D, dtype=dtype, device=device)
    x_ref.copy_(x_shared)

    print(f"[Rank {rank}] Running Helion one-shot allreduce + bias + rmsnorm...")
    result_helion = helion_one_shot_allreduce_bias_rmsnorm(x_shared, bias, weight)

    # Sync before reference to ensure Helion kernel is complete on all ranks
    torch.cuda.synchronize()
    dist.barrier()

    print(f"[Rank {rank}] Running reference...")
    result_ref = reference_one_shot_allreduce_bias_rmsnorm(x_ref, bias, weight)

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

    test(N=128, D=4096, device=device, dtype=torch.float32)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/one_shot_allreduce_bias_rmsnorm.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
