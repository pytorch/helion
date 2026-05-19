"""
All-Gather + FP8 MatMul Fusion Example
========================================
This example extend all_gather_matmul.py to use FP8 inputs.
Each rank holds an FP8 A shard and a full B. The kernel waits for the all-gather to fill the symmetric memory buffer, then computes a local FP8 matmul 
and writes the bfloat16 result to output.
"""

from __future__ import annotations

import os

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from helion.runtime.triton_helpers import triton_wait_signal
from torch.distributed import ProcessGroup



@triton.jit
def _wait_progress_at_idx(progress: tl.tensor, idx: int) -> None:
    triton_wait_signal(progress + idx, 1, 0, "acquire", "gpu", "ld", False)

@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 256, 128],  # M, N, K
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def helion_matmul_w_progress_fp8(
    a: torch.Tensor,  # [M, K] FP8 (full gathered)
    a_shared: torch.Tensor,  # [M//world_size, K] FP8
    scale_a: torch.Tensor,  # [M//world_size, 1] FP32
    b: torch.Tensor,  # [K, N] FP8
    scale_b: torch.Tensor,  # [1, N] FP32
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication with FP8 tensors and tracks progress using Helion.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.bfloat16, device=a.device
    )  # Output buffered as BF16 for performance.
    M_per_rank = a_shared.size(0)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # Initialize accumulator in FP32.
        # Once the progress is filled, we can start doing matmul
        hl.triton_kernel(
            _wait_progress_at_idx,
            args=(
                progress,
                tile_m.begin // (M_per_rank // SPLITS_PER_RANK),
            ),
            output_like=None,
        )
        # Load scales once per tile
        sa = scale_a[tile_m, :] # [tile_m, 1]
        sb = scale_b[:, tile_n]  # [1, tile_n]

        for tile_k in hl.tile(K):
            x_tile = a[tile_m, tile_k] 
            y_tile = b[tile_k, tile_n]
            acc = hl.dot(x_tile, y_tile, acc=acc) 

        # Convert result back to bfloat16
        out[tile_m, tile_n] = (acc * sa * sb).to(torch.bfloat16)

    return out


def copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    group: ProcessGroup,  
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an all-gather operation with progress tracking using symmetric memory.
    Args:
        output (torch.Tensor): The tensor to store the gathered result.
        inp (torch.Tensor): The input tensor that is already in symmetric memory.
        progress (torch.Tensor): A tensor used to track the progress of the all-gather operation.
        group (ProcessGroup): The distributed group for synchronization.
        splits_per_rank (int): The number of splits per rank for progress tracking.
        backend_stream (torch.cuda.Stream, optional): The CUDA stream to use for the operation. If None, a new stream will be created.
    """
    backend_stream = dist._symmetric_memory._get_backend_stream(priority=-1)
    assert inp.is_contiguous(), "Input tensor 'inp' must be contiguous"

    if group is None:
        raise RuntimeError("No symmetric memory group available")

    symm_mem_hdl = dist._symmetric_memory.rendezvous(inp, group=group)
    assert symm_mem_hdl is not None, "Failed to obtain symmetric memory handle"

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    assert inp.numel() % splits_per_rank == 0, "inp.numel must be divisible by splits_per_rank"
    assert progress.numel() >= world_size * splits_per_rank, "progress size is insufficient"

    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, "Mismatch in output shape"
    chunks = output.chunk(world_size * splits_per_rank)

    #symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(backend_stream):
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf, non_blocking=True)
                # Write progress signal
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        #symm_mem_hdl.barrier()

    return backend_stream

def _helion_all_gather_fp8_matmul_runtime(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    world_size: int,
    group_name: ProcessGroup,
    a_out: torch.Tensor | None = None,
    SPLITS_PER_RANK: int = 1, 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an all-gather on a_shared and matrix multiplication using the Helion library.
    """

    symm_mem_group = group_name
    symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared, group=group_name)

    if symm_mem_hdl is None:
        a_shared_symm = dist._symmetric_memory.empty(
            a_shared.shape,
            dtype=a_shared.dtype,
            device=a_shared.device
        )
        a_shared_symm.copy_(a_shared)
        a_shared_symm._is_symmetric_memory = True

        # Try rendezvous again with the symmetric copy
        symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared_symm, group=group_name)
        if symm_mem_hdl is None:
            raise RuntimeError("Failed to get symmetric memory handle after copy")
    else:
        a_shared_symm = a_shared  # already usable

    a_shape = list(a_shared_symm.shape)
    a_shape[0] *= symm_mem_hdl.world_size
    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
    
    progress = torch.zeros(
        symm_mem_hdl.world_size * SPLITS_PER_RANK,
        dtype=torch.uint32,
        device=a_shared_symm.device,
    )

    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared_symm, progress, group_name, SPLITS_PER_RANK
    )
    
    c = helion_matmul_w_progress_fp8(
        a_out,
        a_shared_symm,
        scale_a,
        b,
        scale_b,
        progress,
        SPLITS_PER_RANK=SPLITS_PER_RANK,
        RANK=symm_mem_hdl.rank,
    )
    assert type(c) is torch.Tensor
    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, c

def helion_ag_matmul(a_shared: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, world_size: int, dist_group: ProcessGroup) -> torch.Tensor:
    """Wrapper for helion_all_gather_matmul that returns only the matmul result."""
    a_out, c = _helion_all_gather_fp8_matmul_runtime(
            a_shared,
            b,
            scale_a,
            scale_b,
            world_size,
            dist_group.group_name,
            SPLITS_PER_RANK=1,
        )
    return c


def reference_ag_matmul(a_shared: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, world_size: int, dist_group: ProcessGroup) -> torch.Tensor:
    """Reference implementation using torch.ops.symm_mem.fused_all_gather_matmul."""
    dist_group = dist.group.WORLD
    if dist_group is None:
        raise RuntimeError("No distributed group available")
    ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            a_shared,
            [b],
            scale_a,
            [scale_b],
            gather_dim=0,
            biases=[None],
            result_scales=[None],
            out_dtypes=[torch.bfloat16],
            use_fast_accum=[False],
            group_name=dist_group.group_name,
        )
    return mm_golden[0]


# %%
def test(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    """
    Tests and benchmarks helion_all_gather_matmul against PyTorch's implementation.
    Args:
        M (int): First dimension of the matrix.
        N (int): Second dimension of the matrix.
        K (int): Third dimension of the matrix.
        world_size (int): Number of processes.
        device (torch.device): Device to run the test on.
    """
    M_per_rank = M // world_size
    FP8_DTYPE = torch.float8_e4m3fn
    torch.manual_seed(41)  # deterministic for all ranks
    a_shared = torch.rand(M_per_rank, K, device=device, dtype=torch.bfloat16) * 0.05
    a_shared = a_shared.to(FP8_DTYPE)
    b = (torch.rand(K, N, device=device, dtype=torch.bfloat16) *0.1+ 0.05).T.contiguous().T
    b = b.to(FP8_DTYPE)

    scale_a = torch.rand((M_per_rank, 1), device=device, dtype=torch.float32) * 0.05 + 0.01
    scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

    # clamp scales to prevent overflow when converting to BF16 output
    # in result = (FP8_accumulator * scale_a * scale_b).to(bfloat16)
    min_val=1e-4 
    max_val = 100
    scale_a = scale_a.clamp(min=min_val, max=max_val)
    scale_b = scale_b.clamp(min=min_val, max=max_val)

    #allocation in advance
    a_shared_symm = dist._symmetric_memory.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device
    )
    a_shared_symm.copy_(a_shared)

    run_example(
        helion_ag_matmul,
        reference_ag_matmul,
        (a_shared_symm, b, scale_a, scale_b, world_size, dist.group.WORLD),
        rtol=1e-1,
        atol=1e-1,
    )

def main() -> None:
    """
    Main entry point that initializes the distributed environment and runs the test.
    Sets up the distributed process group, runs the test, and then cleans up.
    """
    _SymmetricMemory.signal_pad_size = 1024 * 1024 * 1024
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(2048, 2560, 8192, world_size, device)
    dist.destroy_process_group()


# %%
if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 examples/distributed/fp8_scaled_all_gather_matmul.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
