"""
MatMul + Reduce-Scatter Fusion Example
=======================================
This example demonstrates how to implement a fused matrix multiplication followed by
reduce-scatter using Helion and PyTorch's distributed capabilities. It includes a Helion
kernel demonstrating how to use symm_mem_sync Triton kernel for cross-device synchronization
and torch.ops.symm_mem.get_remote_tensors for accessing symmetric memory tensors on peer devices.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import os
from typing import Iterator

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from examples.distributed.utils import distributed_benchmark
from examples.distributed.utils import symm_mem_sync

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


@contextmanager
def ieee_matmul_precision() -> Iterator[None]:
    """Context manager to temporarily set IEEE precision for matmul operations."""
    old_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_precision


def compute_signal_pad_size(
    M: int, N: int, world_size: int, min_block_size: int = 32
) -> int:
    """Compute the required signal pad size based on tile counts.

    The kernel uses 2 barriers per tile (write_barrier_id and read_barrier_id),
    and each barrier uses world_size slots. We use a conservative estimate
    based on the minimum expected block size.

    Args:
        M: M dimension of the output matrix
        N: N dimension of the output matrix
        world_size: Number of ranks in the distributed group
        min_block_size: Minimum expected block size (smaller = more tiles = more slots needed)

    Returns:
        Required signal pad size in bytes
    """
    # Conservative tile count estimates using minimum block size
    tile_m_count = (M + min_block_size - 1) // min_block_size
    tile_n_count = (N + min_block_size - 1) // min_block_size

    # Each barrier uses world_size slots, we have 2 barriers per tile
    required_slots = 2 * tile_m_count * tile_n_count * world_size

    # Each slot is a uint32 (4 bytes)
    return required_slots * 4


def reference_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation using separate PyTorch operations.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    # Compute local matmul with IEEE precision to match kernel's dot_precision="ieee"
    with ieee_matmul_precision():
        c = torch.mm(a.to(torch.float32), b.to(torch.float32)).to(a.dtype)

    # Reduce-scatter along dimension 0
    world_size = dist.get_world_size(group)
    M = c.shape[0]
    M_scatter = M // world_size

    # Create output tensor
    output = torch.empty(M_scatter, c.shape[1], dtype=c.dtype, device=c.device)

    # Sync all ranks before the collective operation
    # This is needed because the baseline function may be called at different times
    # by different ranks during autotuning
    dist.barrier(group=group)

    # Perform reduce-scatter
    dist.reduce_scatter_tensor(output, c, group=group)

    return output


def create_benchmark_inputs() -> tuple:
    """Create benchmark inputs for matmul_reduce_scatter_kernel.

    This function is called in each worker subprocess after dist.init_process_group().
    It should return a tuple of kernel arguments.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    assert group is not None

    M, N, K = 256, 384, 512
    dtype = torch.float32

    torch.manual_seed(42 + rank)
    a = torch.randn(M, K, dtype=dtype, device=device)
    torch.manual_seed(42)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # Set signal pad size based on tile counts before allocating symmetric memory
    world_size = dist.get_world_size(group)
    signal_pad_size = compute_signal_pad_size(M, N, world_size)
    symm_mem.set_signal_pad_size(signal_pad_size)

    # Setup symmetric memory
    symm_mem_buffer = symm_mem.empty(M, N, dtype=dtype, device=device)
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

    return (
        a,
        b,
        symm_mem_buffer,
        symm_mem_hdl.signal_pad_ptrs_dev,
        symm_mem_hdl.rank,
        symm_mem_hdl.world_size,
        group.group_name,
    )


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64, 32],  # M, N, K
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    autotune_benchmark_fn=partial(
        distributed_benchmark, inputs_fn=create_benchmark_inputs
    ),
    autotune_baseline_fn=lambda a,
    b,
    buf,
    sig,
    rank,
    ws,
    gn: reference_matmul_reduce_scatter(a, b),
    dot_precision="ieee",
)
def matmul_reduce_scatter_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    symm_mem_buffer: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    """
    Fused MatMul + Reduce-Scatter kernel.
    """
    M, K = a.size()
    K2, N = b.size()
    M_scatter = M // WORLD_SIZE  # type: ignore[unsupported-operation]

    # Output is only M/world_size rows per rank
    output = torch.empty([M_scatter, N], dtype=a.dtype, device=a.device)

    # Get remote buffers from all ranks
    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, GROUP_NAME)

    # Compute which rows this rank is responsible for in the scatter
    scatter_start = RANK * M_scatter  # type: ignore[unsupported-operation]
    scatter_end = scatter_start + M_scatter  # type: ignore[unsupported-operation]

    # Tile over (M, N) for the full GEMM
    for tile_m, tile_n in hl.tile([M, N]):
        # Step 1: Compute local GEMM tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])

        # Step 2: Store to this rank's symmetric memory buffer
        symm_mem_buffer[tile_m, tile_n] = acc.to(a.dtype)

        # Step 3: Sync with hasPreviousMemAccess=True hasSubsequentMemAccess=True
        # - release fence: ensures our write to symm_mem_buffer is visible to other ranks
        # - acquire fence: ensures we see other ranks' writes to their buffers
        # Compute unique barrier ID from tile indices (no hardcoded constants)
        num_n_tiles = tile_n.count
        write_barrier_id = tile_m.id * num_n_tiles + tile_n.id
        hl.triton_kernel(
            symm_mem_sync,
            args=(
                signal_pad_ptrs,
                write_barrier_id,
                RANK,
                WORLD_SIZE,
                True,
                True,
            ),
            output_like=None,
        )

        # Step 4: Conditional reduce-scatter - check if this tile falls within our scatter range
        if tile_m.begin >= scatter_start and tile_m.begin < scatter_end:  # type: ignore[unsupported-operation]
            # This tile belongs to us - reduce from all ranks
            acc_reduce = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for remote_buffer in buffer_tuple:
                acc_reduce = acc_reduce + remote_buffer[tile_m, tile_n].to(
                    torch.float32
                )

            # Write to output at local offset
            output[tile_m.index - scatter_start, tile_n] = acc_reduce.to(a.dtype)  # type: ignore[unsupported-operation]

        # Step 5: Final sync
        # Use offset by total tiles to ensure unique barrier IDs
        total_tiles = tile_m.count * num_n_tiles
        read_barrier_id = total_tiles + write_barrier_id
        hl.triton_kernel(
            symm_mem_sync,
            args=(
                signal_pad_ptrs,
                read_barrier_id,
                RANK,
                WORLD_SIZE,
                True,
                False,
            ),
            output_like=None,
        )

    return output


def helion_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper that sets up symmetric memory and calls the Helion kernel.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"

    world_size = dist.get_world_size(group)

    assert M % world_size == 0, (
        f"M dimension ({M}) must be divisible by world_size ({world_size})"
    )

    # Set signal pad size based on tile counts before allocating symmetric memory
    signal_pad_size = compute_signal_pad_size(M, N, world_size)
    symm_mem.set_signal_pad_size(signal_pad_size)

    # Create symmetric memory buffer for the full C matrix
    symm_mem_buffer = symm_mem.empty(M, N, dtype=a.dtype, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

    return matmul_reduce_scatter_kernel(
        a,
        b,
        symm_mem_buffer,
        symm_mem_hdl.signal_pad_ptrs_dev,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        GROUP_NAME=group.group_name,
    )


def test(M: int, N: int, K: int, device: torch.device, dtype: torch.dtype) -> None:
    """Test the Helion implementation against the reference."""
    rank = dist.get_rank()

    # Each rank has the same random seed for reproducibility
    torch.manual_seed(42 + rank)
    a = torch.randn(M, K, dtype=dtype, device=device)

    # Weight matrix is the same across all ranks
    torch.manual_seed(42)
    b = torch.randn(K, N, dtype=dtype, device=device)

    run_example(
        helion_matmul_reduce_scatter,
        reference_matmul_reduce_scatter,
        (a, b),
        rtol=1e-3,
        atol=1e-3,
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

    # Test with M divisible by world_size
    # M=512, K=1024, N=768 with 4 GPUs -> output is 128x768 per rank
    test(M=512, N=768, K=1024, device=device, dtype=torch.float32)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    examples/distributed/matmul_reduce_scatter.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
