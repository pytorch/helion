"""
All-Gather FP8 Scaled Matrix Multiplication Example
====================================================
This example extends the all_gather_matmul example to use FP8 inputs with
per-row and per-column scaling.  Each rank holds an FP8 shard of A in
symmetric memory; a copy-engine all-gather streams those shards into a
contiguous buffer while the compute kernel overlaps the gather with FP8
GEMM tiles.

Each output element is:

    out[i, j] = (scale_a[i] * scale_b[j] * (A_gathered @ B)[i, j]).to(bfloat16)
"""

from __future__ import annotations

import functools
import os

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import assert_close_with_mismatch_tolerance
from helion._testing import run_example
import helion.language as hl
from helion.runtime.triton_helpers import triton_wait_signal

tolerance = {
    "atol": 5e-1,
    "rtol": 5e-1,
    "max_mismatch_pct": 1e-3,
}


@triton.jit
def _wait_progress_at_idx(progress: tl.tensor, idx: int) -> None:
    triton_wait_signal(progress + idx, 1, 0, "acquire", "gpu", "ld", False)


def copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,
    progress: torch.Tensor,
    splits_per_rank: int,
) -> torch.cuda.Stream:
    """
    All-gather ``inp`` (a symmetric tensor) into ``output``, writing a progress
    counter after each split so the compute kernel can start as soon as the
    rows it needs are ready.
    """
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    assert inp.is_contiguous()
    symm_mem_group = dist.group.WORLD
    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")
    symm_mem_hdl = symm_mem.rendezvous(inp, group=symm_mem_group)
    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    assert inp.numel() % splits_per_rank == 0
    assert progress.numel() >= world_size * splits_per_rank
    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape
    chunks = output.chunk(world_size * splits_per_rank)
    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(backend_stream):
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank,
                    chunks[0].shape,
                    inp.dtype,
                    chunks[0].numel() * split_id,
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        symm_mem_hdl.barrier()
    return backend_stream


@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 128, 64],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
    autotune_baseline_accuracy_check_fn=functools.partial(
        assert_close_with_mismatch_tolerance,
        **tolerance,
    ),
)
def fp8_ag_scaled_matmul_kernel(
    a: torch.Tensor,  # [M, K] float8_e4m3fn, full gathered buffer
    b: torch.Tensor,  # [K, N] float8_e4m3fn
    scale_a: torch.Tensor,  # [M, 1] float32
    scale_b: torch.Tensor,  # [1, N] float32
    progress: torch.Tensor,
    SPLITS_PER_RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    RANK: hl.constexpr,
) -> torch.Tensor:
    """
    FP8 GEMM with per-row/column scaling, overlapped with a copy-engine all-gather.

    Before processing each tile_m block the kernel waits on the progress counter
    that the copy engine writes when the corresponding rows have been transferred.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    M_per_rank = M // WORLD_SIZE  # type: ignore[unsupported-operation]

    out = torch.empty([M, N], dtype=torch.bfloat16, device=a.device)

    for tile_m, tile_n in hl.tile([M, N]):
        # Wait for the copy engine to have gathered the rows we need
        hl.triton_kernel(
            _wait_progress_at_idx,
            args=(progress, tile_m.begin // (M_per_rank // SPLITS_PER_RANK)),  # type: ignore[unsupported-operation]
            output_like=None,
        )
        # FP8 GEMM, accumulate in FP32
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
        # Apply per-row and per-column scales
        acc = (
            acc
            * scale_a[tile_m, :].to(torch.float32)
            * scale_b[:, tile_n].to(torch.float32)
        )
        out[tile_m, tile_n] = acc.to(torch.bfloat16)

    return out


def helion_all_gather_scaled_matmul(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    splits_per_rank: int = 1,
) -> torch.Tensor:
    """
    All-gather FP8 A shards, then compute a scaled FP8 GEMM.

    Args:
        a_shared: Local FP8 A shard ``[M//world_size, K]`` in symmetric memory.
        b: FP8 weight matrix ``[K, N]``.
        scale_a: Per-row scale for A ``[M, 1]`` float32.
        scale_b: Per-column scale for B ``[1, N]`` float32.
        splits_per_rank: Number of progress-tracking splits per rank.
    """
    symm_mem_group = dist.group.WORLD
    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")
    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=symm_mem_group)
    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    M_per_rank, K = a_shared.shape
    M = M_per_rank * world_size

    a_out = torch.empty(M, K, dtype=a_shared.dtype, device=a_shared.device)
    progress = torch.zeros(
        world_size * splits_per_rank, dtype=torch.uint32, device=a_shared.device
    )

    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, splits_per_rank
    )
    out = fp8_ag_scaled_matmul_kernel(
        a_out,
        b,
        scale_a,
        scale_b,
        progress,
        SPLITS_PER_RANK=splits_per_rank,
        WORLD_SIZE=world_size,
        RANK=rank,
    )
    torch.cuda.current_stream().wait_stream(backend_stream)
    return out


def reference_all_gather_scaled_matmul(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Reference: all-gather A, then apply torch._scaled_mm."""
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("No distributed group available")
    world_size = dist.get_world_size(group)
    M_per_rank, K = a_shared.shape
    M = M_per_rank * world_size

    a_full = torch.empty(M, K, dtype=a_shared.dtype, device=a_shared.device)
    dist.all_gather_into_tensor(a_full, a_shared, group=group)

    return torch._scaled_mm(
        a_full, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
    )


def test(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    """Test the Helion all-gather scaled matmul against the reference."""
    rank = dist.get_rank()

    torch.manual_seed(42 + rank)
    a_shard_fp32 = torch.randn(M // world_size, K, device=device)
    a_shared = symm_mem.empty(
        M // world_size, K, dtype=torch.float8_e4m3fn, device=device
    )
    a_shared.copy_(a_shard_fp32.to(torch.float8_e4m3fn))

    torch.manual_seed(42)
    b_fp32 = torch.randn(K, N, device=device)
    b = b_fp32.to(torch.float8_e4m3fn).T.contiguous().T

    scale_a = torch.rand(M, 1, device=device)
    scale_b = torch.rand(1, N, device=device)

    run_example(
        helion_all_gather_scaled_matmul,
        reference_all_gather_scaled_matmul,
        (a_shared, b, scale_a, scale_b),
        **tolerance,
    )


def main() -> None:
    _SymmetricMemory.signal_pad_size = 1024 * 1024 * 1024
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    test(M=4096, N=6656, K=16384, world_size=world_size, device=device)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \\
    --nproc-per-node 4 \\
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \\
    examples/distributed/fp8_all_gather_matmul.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
