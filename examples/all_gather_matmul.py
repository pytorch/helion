"""
All-Gather Matrix Multiplication Example
========================================
This example demonstrates how to implement an all-gather operation followed by matrix multiplication
using Helion and PyTorch's distributed capabilities. It includes progress tracking using symmetric memory
and a Helion kernel optimized for multi-process runs.
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
import helion.language as hl


# %%
def copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an all-gather operation with progress tracking using symmetric memory.
    Args:
        output (torch.Tensor): The output tensor to store the gathered results.
        inp (torch.Tensor): The input tensor to be gathered (must be a symmetric tensor).
        progress (torch.Tensor): Tensor used to track progress of the operation.
        splits_per_rank (int): Number of splits per rank.
        backend_stream (torch.cuda.Stream, optional): CUDA stream for backend operations.
    Returns:
        torch.cuda.Stream: The CUDA stream used for the operation.
    """
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    assert inp.is_contiguous()
    symm_mem_group = dist.group.WORLD
    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")
    symm_mem_hdl = symm_mem.rendezvous(inp, group=symm_mem_group)
    assert symm_mem_hdl is not None
    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    assert inp.numel() % splits_per_rank == 0
    assert progress.numel() >= world_size * splits_per_rank
    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, (list(output.shape), output_shape)
    chunks = output.chunk(world_size * splits_per_rank)
    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(backend_stream):
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
                # cuStreamWriteValue32 issues a system level fence before the write
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        symm_mem_hdl.barrier()
    return backend_stream


# %%
@helion.jit(
    config=helion.Config(
        block_sizes=[128, 256, 64],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def helion_matmul_w_progress(
    a: torch.Tensor,
    a_shared: torch.Tensor,
    b: torch.Tensor,
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication with progress tracking.
    Args:
        a (torch.Tensor): First input tensor for matrix multiplication.
        a_shared (torch.Tensor): Shared tensor across ranks.
        b (torch.Tensor): Second input tensor for matrix multiplication.
        progress (torch.Tensor): Tensor used to track progress of the operation.
        SPLITS_PER_RANK (int): Number of splits per rank.
        RANK (int): Current process rank.
    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.promote_types(a.dtype, b.dtype), device=a.device
    )
    M_per_rank = a_shared.size(0)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        hl.wait(
            progress,
            [
                tile_m.begin // (M_per_rank // SPLITS_PER_RANK),
            ],
            signal=1,
        )
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


# %%
def helion_all_gather_matmul(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    **kwargs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combines all-gather and matrix multiplication operations.
    Args:
        a_shared (torch.Tensor): Shared tensor across ranks to be gathered.
        b (torch.Tensor): Second input tensor for matrix multiplication.
        a_out (torch.Tensor, optional): Optional output tensor for the gathered results.
        progress (torch.Tensor, optional): Optional tensor used to track progress of the operation.
        **kwargs: Additional keyword arguments including splits_per_rank.
    Returns:
        tuple: A tuple containing:
            - The gathered tensor.
            - The result of the matrix multiplication.
    """
    configs = {
        "SPLITS_PER_RANK": kwargs.get("splits_per_rank", 1),
    }
    symm_mem_group = dist.group.WORLD
    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")
    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=symm_mem_group)
    a_shape = list(a_shared.shape)
    a_shape[0] *= symm_mem_hdl.world_size
    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = symm_mem_hdl.world_size
    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
    if progress is None:
        progress = torch.zeros(
            symm_mem_hdl.world_size * configs["SPLITS_PER_RANK"],
            dtype=torch.uint32,
            device=a_shared.device,
        )
    else:
        progress.fill_(0)  # Reset progress to 0.
    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, configs["SPLITS_PER_RANK"]
    )
    c = helion_matmul_w_progress(
        a_out,
        a_shared,
        b,
        progress,
        SPLITS_PER_RANK=configs["SPLITS_PER_RANK"],
        RANK=configs["RANK"],
    )
    assert type(c) is torch.Tensor
    torch.cuda.current_stream().wait_stream(backend_stream)
    return a_out, c


# %%
def test(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    """
    Tests the helion_all_gather_matmul function against PyTorch's implementation.
    Args:
        M (int): First dimension of the matrix.
        N (int): Second dimension of the matrix.
        K (int): Third dimension of the matrix.
        world_size (int): Number of processes.
        device (torch.device): Device to run the test on.
    """
    a_shared = symm_mem.empty(
        M // world_size, K, dtype=torch.bfloat16, device=device
    ).normal_()
    b = torch.randn((K, N), device=DEVICE, dtype=torch.bfloat16).T.contiguous().T
    a_out, c = helion_all_gather_matmul(a_shared, b)
    golden_a = a_shared.clone()
    dist_group = dist.group.WORLD
    if dist_group is None:
        raise RuntimeError("No distributed group available")
    ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(  # pyright: ignore[reportCallIssue]
        golden_a, [b], gather_dim=0, group_name=dist_group.group_name
    )
    torch.testing.assert_close(c, mm_golden[0], rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(a_out, ag_golden)


# %%
def main() -> None:
    """
    Main entry point that initializes the distributed environment and runs the test.
    Sets up the distributed process group, runs the test, and then cleans up.
    """
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(4096, 6656, 16384, world_size, device)
    dist.destroy_process_group()


# %%
if __name__ == "__main__":
    """
    Run with:
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 examples/all_gather_matmul.py
    """
    # TODO(adam-smnk): generalize to XPU
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
