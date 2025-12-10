"""
One-Shot All-Reduce + Bias + RMS Norm Fusion
=============================================
Uses hl.triton_kernel to call symm_mem_sync function from utils.py,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.

Maps to: kraken/fused/one_shot_all_reduce_bias_rms_norm.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

from examples.distributed.utils import symm_mem_sync  # noqa: F401


@helion.jit(
    config=helion.Config(
        block_sizes=[8],  # Process multiple rows at a time (fewer sync IDs)
        num_warps=8,
    ),
    static_shapes=True,
)
def one_shot_allreduce_bias_rmsnorm_kernel(
    x: torch.Tensor,  # Regular input tensor (not in symmetric memory)
    symm_mem_buffer: torch.Tensor,  # Symmetric memory buffer for this rank
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

    Matches Kraken's pattern:
        1. Copy input x to symmetric memory buffer
        2. Barrier sync (Pattern 1: release+acquire)
        3. All-reduce: acc = bias + sum(buffer from all ranks)
        4. RMS Norm: y = acc * rsqrt(mean(acc^2) + eps) * weight
        5. Final barrier sync
    """
    N, D = x.size()
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote buffers from all ranks (views into each rank's symm_mem_buffer)
    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, GROUP_NAME)

    for tile_n in hl.tile(N):
        # Step 1: Copy input x to our symmetric memory buffer
        symm_mem_buffer[tile_n, :] = x[tile_n, :]

        # Step 2: Sync with Pattern 1 (hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
        # - release fence: ensures our write to symm_mem_buffer is visible to other ranks
        # - acquire fence: ensures we see other ranks' writes to their buffers
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, tile_n.id, RANK, world_size, True, True),
            output_like=None,
        )

        # Step 3: All-reduce + bias: acc = bias + sum(buffer from all ranks)
        # Initialize acc with the right shape by broadcasting bias
        acc = symm_mem_buffer[tile_n, :].to(torch.float32) * 0.0 + bias[None, :].to(
            torch.float32
        )
        for remote_buffer in buffer_tuple:
            acc = acc + remote_buffer[tile_n, :].to(torch.float32)

        # Step 4: RMS Norm: y = acc * rsqrt(mean(acc^2) + eps) * weight
        variance = torch.mean(acc * acc, dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + EPS)
        normalized = acc * rstd
        output[tile_n, :] = (normalized * weight[None, :].to(torch.float32)).to(x.dtype)

        # Step 5: Final sync (Pattern 2: release only)
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, tile_n.id, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_one_shot_allreduce_bias_rmsnorm(
    x: torch.Tensor,  # Regular input tensor
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Wrapper that sets up symmetric memory and calls the Helion kernel.

    Unlike the previous version, this takes a regular tensor x and copies
    it to symmetric memory within the kernel (matching Kraken's pattern).
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    N, D = x.shape

    # Allocate symmetric memory buffer (separate from input)
    symm_mem_buffer = symm_mem.empty(N, D, dtype=x.dtype, device=x.device)
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

    output = torch.empty((N, D), dtype=x.dtype, device=x.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=x.device)

    return one_shot_allreduce_bias_rmsnorm_kernel(
        x,  # Regular input tensor
        symm_mem_buffer,  # Symmetric memory buffer
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
    """Test the Helion implementation against the reference using run_example."""
    rank = dist.get_rank()

    # Create regular input tensor (not symmetric memory - kernel will copy it)
    torch.manual_seed(42 + rank)
    x = torch.randn(N, D, dtype=dtype, device=device)

    # Bias and weight are the same across ranks (inference use case)
    torch.manual_seed(42)
    bias = torch.randn(D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)

    print(f"[Rank {rank}] Running correctness test and benchmark...")
    run_example(
        helion_one_shot_allreduce_bias_rmsnorm,
        reference_one_shot_allreduce_bias_rmsnorm,
        (x, bias, weight),
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"[Rank {rank}] Test passed!")


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
    examples/distributed/one_shot_allreduce_bias_rmsnorm.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
