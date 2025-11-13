"""
Fused GEMM + One-Shot All-Reduce - Simplified Version
======================================================
Uses hl.triton_kernel to call the Kraken symm_mem_sync function,
and torch.ops.symm_mem.get_remote_tensors for simplified buffer management.
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
def _gemm_one_shot_all_reduce_kernel(
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
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    # Specialize world_size to make it a compile-time constant
    world_size = hl.specialize(world_size_tensor.size(0))

    # Get remote tensors inside the kernel using the new op
    buf_tuple = torch.ops.symm_mem.get_remote_tensors(local_buf, GROUP_NAME)

    # Compute local GEMM and write to shared buffer
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        local_buf[tile_m, tile_n] = acc

        # Synchronize using hl.triton_kernel to call Kraken's symm_mem_sync
        sync_id = tile_m.id * helion.cdiv(N, 64) + tile_n.id
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id, RANK, world_size, False, True),
            output_like=None,
        )

        # All-reduce: sum results from all ranks
        reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for buf in buf_tuple:
            reduced += buf[tile_m, tile_n]
        output[tile_m, tile_n] = reduced

        # Final synchronization
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_gemm_one_shot_all_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size_k: int = 64,
) -> torch.Tensor:
    """Runs the fused GEMM + one-shot all-reduce kernel."""

    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("Example currently supports float32 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")

    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    M, K = a.shape
    _, N = b.shape

    # Allocate symmetric memory buffer directly using symm_mem.empty
    local_buf = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)

    # Get signal pad for synchronization
    symm_mem_hdl = symm_mem.rendezvous(local_buf, group.group_name)

    output = torch.empty((M, N), dtype=torch.float32, device=a.device)

    # Create a tensor to carry world_size for hl.specialize
    world_size_tensor = torch.empty(symm_mem_hdl.world_size, device=a.device)

    return _gemm_one_shot_all_reduce_kernel(
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


def _reference_gemm_one_shot_all_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.matmul(a, b)
    dist.all_reduce(out)
    return out


def test(M: int, N: int, K: int, device: torch.device) -> None:
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)

    helion_result = helion_gemm_one_shot_all_reduce(a, b)
    reference = _reference_gemm_one_shot_all_reduce(a, b)

    torch.testing.assert_close(helion_result, reference, rtol=1e-1, atol=1e-1)
    print(f"Rank {dist.get_rank()}: Test passed!")


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

    test(512, 256, 128, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python examples/gemm_one_shot_all_reduce.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
