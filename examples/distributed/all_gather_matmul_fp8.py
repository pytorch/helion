# %%
# FP8 All-Gather Matrix Multiplication Example
# ===========================================
# Same as all_gather_matmul.py but using FP8 inputs with FP32 accumulation and FP16 output

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
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        symm_mem_hdl.barrier()
    return backend_stream

# %%
@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 256, 64],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def helion_matmul_w_progress_fp8(
    a: torch.Tensor,      # FP8 input
    a_shared: torch.Tensor,  # FP8 gathered input
    b: torch.Tensor,      # FP8 input
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.promote_types(torch.float16, torch.float16), device=a.device
    )
    M_per_rank = a_shared.size(0)
    for tile_m, tile_n in hl.tile([M, N]):
        # Accumulate in FP32
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        hl.wait(
            progress,
            [
                tile_m.begin // (M_per_rank // SPLITS_PER_RANK),
            ],
            signal=1,
        )
        for tile_k in hl.tile(K):
            # Cast FP8 -> FP32 for accumulation
            acc = torch.addmm(acc, a[tile_m, tile_k].to(torch.float32), b[tile_k, tile_n].to(torch.float32))
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out

# %%
def helion_all_gather_matmul_fp8(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    **kwargs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        progress.fill_(0) # Reset progress to 0.
    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, configs["SPLITS_PER_RANK"]
    )
    c = helion_matmul_w_progress_fp8(
        a_out,
        a_shared,
        b,
        progress,
        SPLITS_PER_RANK=configs["SPLITS_PER_RANK"],
        RANK=configs["RANK"],
    )
    torch.cuda.current_stream().wait_stream(backend_stream)
    return a_out, c

# %%
def test_fp8(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    # Create FP8 symmetric input
    a_shared = symm_mem.empty(
        M // world_size, K, dtype=torch.float8_e4m3fn, device=device
    ).normal_()
    b = torch.randn((K, N), device=device, dtype=torch.float8_e4m3fn).T.contiguous().T
    a_out, c = helion_all_gather_matmul_fp8(a_shared, b)
    # Reference FP16 accumulation for correctness
    golden_a = a_shared.to(torch.float32).clone()
    golden_b = b.to(torch.float32).clone()
    mm_golden = golden_a @ golden_b
    # Convert Helion output to FP32 for comparison
    torch.testing.assert_close(c.to(torch.float32), mm_golden, rtol=1e-1, atol=1e-1)

# %%
def main_fp8() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test_fp8(4096, 6656, 16384, world_size, device)
    dist.destroy_process_group()

# %%
if __name__ == "__main__":
    print("Running FP8 All-Gather MatMul Example")
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main_fp8()
