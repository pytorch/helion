"""
One-Shot All-Reduce + Bias + RMS Norm Fusion Example
=====================================================
This example demonstrates how to implement a fused one-shot all-reduce with bias
addition and RMS normalization using Helion and PyTorch's distributed capabilities.
It includes a Helion kernel demonstrating how to use symm_mem_sync Triton kernel for
cross-device synchronization and torch.ops.symm_mem.get_remote_tensors for accessing symmetric
memory tensors on peer devices.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from examples.distributed.utils import symm_mem_sync

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@helion.jit(
    config=helion.Config(
        block_sizes=[8],
        num_warps=8,
    ),
    static_shapes=True,
)
def one_shot_allreduce_bias_rmsnorm_kernel(
    x: torch.Tensor,
    symm_mem_buffer: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    EPS: hl.constexpr,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    """
    Fused one-shot all-reduce + bias addition + RMS normalization.
    """
    N, D = x.size()
    output = torch.empty_like(x)

    # Get remote buffers from all ranks (views into each rank's symm_mem_buffer)
    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, GROUP_NAME)

    for tile_n in hl.tile(N):
        # Step 1: Copy input x to our symmetric memory buffer
        symm_mem_buffer[tile_n, :] = x[tile_n, :]

        # Step 2: Sync with hasPreviousMemAccess=True hasSubsequentMemAccess=True
        # - release fence: ensures our write to symm_mem_buffer is visible to other ranks
        # - acquire fence: ensures we see other ranks' writes to their buffers
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, tile_n.id, RANK, WORLD_SIZE, True, True),
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
        rstd = torch.rsqrt(variance + EPS)  # type: ignore[unsupported-operation]
        normalized = acc * rstd
        output[tile_n, :] = (normalized * weight[None, :].to(torch.float32)).to(x.dtype)

        # Step 5: Final sync (release only)
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, tile_n.id, RANK, WORLD_SIZE, True, False),
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
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    N, D = x.shape

    symm_mem_buffer = symm_mem.empty(N, D, dtype=x.dtype, device=x.device)
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

    return one_shot_allreduce_bias_rmsnorm_kernel(
        x,
        symm_mem_buffer,
        bias,
        weight,
        symm_mem_hdl.signal_pad_ptrs_dev,
        EPS=eps,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        GROUP_NAME=group.group_name,
    )


def reference_one_shot_allreduce_bias_rmsnorm(
    x: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
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

    torch.manual_seed(42 + rank)
    x = torch.randn(N, D, dtype=dtype, device=device)

    torch.manual_seed(42)
    bias = torch.randn(D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)

    run_example(
        helion_one_shot_allreduce_bias_rmsnorm,
        reference_one_shot_allreduce_bias_rmsnorm,
        (x, bias, weight),
        rtol=1e-4,
        atol=1e-4,
    )


def main() -> None:
    symm_mem.set_backend("NVSHMEM")
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    symm_mem.enable_symm_mem_for_group(
        dist.group.WORLD.group_name  # type: ignore[missing-attribute]
    )

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
