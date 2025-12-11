"""
Fused GEMM + One-Shot All-Reduce Example
=========================================
This example demonstrates how to implement a fused matrix multiplication (GEMM)
with a one-shot all-reduce operation using Helion and PyTorch's distributed
capabilities. It showcases:

The kernel performs:
1. Local GEMM: C_local = A @ B (each rank computes full result with its local A)
2. Barrier sync: Ensures all ranks have written their local results
3. All-reduce: Sum results from all ranks to get final output
4. Final sync: Ensures reads are complete before buffers can be reused
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

from examples.distributed.utils import dev_array_to_tensor  # noqa: E402
from examples.distributed.utils import symm_mem_sync  # noqa: E402


@helion.jit(
    config=helion.Config(
        block_sizes=[64, 64, 64],  # M, N, K block sizes
        num_warps=8,
        num_stages=3,
    ),
    static_shapes=True,
)
def gemm_one_shot_all_reduce_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    local_buf: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    RANK: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    output = torch.empty((M, N), dtype=torch.float32, device=a.device)
    world_size = hl.specialize(signal_pad_ptrs.size(0))
    buf_tuple = torch.ops.symm_mem.get_remote_tensors(local_buf, GROUP_NAME)

    # Compute local GEMM and write to shared buffer
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        local_buf[tile_m, tile_n] = acc

        # Barrier: signal completion and wait (with acquire) to see all ranks' writes
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

        # Barrier: ensure reads complete (with release) before buffers can be rewritten
        hl.triton_kernel(
            "symm_mem_sync",
            args=(signal_pad_ptrs, sync_id, RANK, world_size, True, False),
            output_like=None,
        )

    return output


def helion_gemm_one_shot_all_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Runs the fused GEMM + one-shot all-reduce kernel.

    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N)
    """
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("Example currently supports float32 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")

    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    M, K = a.shape
    _, N = b.shape

    # Allocate symmetric memory buffer for local GEMM results
    local_buf = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(local_buf, group.group_name)

    # Convert signal_pad_ptrs_dev to tensor
    signal_pad_ptrs = dev_array_to_tensor(
        symm_mem_hdl.signal_pad_ptrs_dev,
        (symm_mem_hdl.world_size,),
        dtype=torch.uint64,
        device=a.device,
    )

    result = gemm_one_shot_all_reduce_kernel(
        a,
        b,
        local_buf,
        signal_pad_ptrs,
        RANK=symm_mem_hdl.rank,
        GROUP_NAME=group.group_name,
    )
    symm_mem_hdl.barrier()
    return result


def reference_gemm_one_shot_all_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation using symm_mem one-shot all-reduce.

    Uses the symmetric memory one-shot primitive for consistency with
    helion version's collective operations.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    # Compute local GEMM
    local_result = torch.matmul(a, b)

    # Allocate symmetric memory and do rendezvous
    result_shared = symm_mem.empty(
        local_result.shape,
        dtype=local_result.dtype,
        device=local_result.device,
    )
    symm_mem_hdl = symm_mem.rendezvous(result_shared, group.group_name)
    result_shared.copy_(local_result)

    # Use symmetric memory one-shot all-reduce
    result = torch.ops.symm_mem.one_shot_all_reduce(
        result_shared, "sum", group.group_name
    )
    symm_mem_hdl.barrier()
    return result


def check(M: int, N: int, K: int, device: torch.device) -> None:
    """
    Test the Helion GEMM + all-reduce against PyTorch's reference implementation.
    Args:
        M: First matrix dimension
        N: Second matrix dimension
        K: Inner dimension
        device: CUDA device to run the test on
    """
    dist_group = dist.group.WORLD
    assert dist_group is not None
    
    # Enable symmetric memory for the group
    symm_mem.enable_symm_mem_for_group(dist_group.group_name)

    # Create input matrices
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)

    run_example(
        helion_gemm_one_shot_all_reduce,
        reference_gemm_one_shot_all_reduce,
        (a, b),
        kernel_name="helion_gemm_one_shot_all_reduce",
        baseline_name="dist_all_reduce",
    )


def main() -> None:
    # Only NVSHMEM backend implements get_remote_tensors for now
    symm_mem.set_backend("NVSHMEM")
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    check(512, 256, 128, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/distributed/gemm_one_shot_all_reduce.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
