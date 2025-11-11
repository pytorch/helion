"""
Fused GEMM + One-Shot All-Reduce Example
=======================================
This example demonstrates how to fuse a matrix multiplication with a symmetric-memory
one-shot all-reduce using Helion. Each rank computes its local GEMM tile, stages the
result in symmetric memory, and then performs a cross-device reduction by coordinating
through signal pads.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load_inline

import helion
from helion._testing import DEVICE
import helion.language as hl


_FROM_BLOB_CPP = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


at::Tensor from_blob(uint64_t data_ptr, c10::IntArrayRef sizes, py::object dtype) {

    at::Tensor tensor = at::for_blob((void*)data_ptr, sizes)
             .deleter([](void *ptr) {
               ;
             })
             .options(at::device(at::kCUDA).dtype(((THPDtype*)dtype.ptr())->scalar_type))
             .make_tensor();

    return tensor;
}
"""

_FROM_BLOB_MOD = load_inline(
    name="helion_symm_mem_from_blob",
    cpp_sources=_FROM_BLOB_CPP,
    functions=["from_blob"],
    with_cuda=True,
)


def _dev_array_to_tensor(
    dev_array_ptr: int, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Creates a tensor view over a device array of pointers."""

    return _FROM_BLOB_MOD.from_blob(dev_array_ptr, shape, dtype)


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
    buf_tuple: tuple[torch.Tensor, ...],
    local_signal_pad: torch.Tensor,
    signal_pad_addrs: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_K: hl.constexpr,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
) -> torch.Tensor:
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    local_buf = buf_tuple[RANK]

    # Compute local GEMM and write to shared buffer
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])

        local_buf[tile_m, tile_n] = acc

        # Synchronization - ensure all writes are visible before reads
        ptr_tile = signal_pad_addrs[:]
        stack_signalpad = hl.stacktensor_like(local_signal_pad, ptr_tile)

        # Use tile indices for synchronization
        sync_id = tile_m.id * helion.cdiv(N, 64) + tile_n.id

        hl.signal(
            stack_signalpad,
            [sync_id, RANK],
            signal=1,
            wait_for=0,
            scope="sys",
            hasPreviousMemAccess=True,
        )

        for world in hl.tile(WORLD_SIZE, block_size=WORLD_SIZE):
            hl.wait(
                local_signal_pad,
                [sync_id, world],
                signal=1,
                update=0,
                scope="sys",
                hasSubsequentMemAccess=True,  # Changed: reads will follow
            )

        # Perform reduction
        reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for shard in buf_tuple:
            reduced += shard[tile_m, tile_n]
        output[tile_m, tile_n] = reduced

    return output


def _prepare_signal_pad(
    symm_mem_hdl,
    tiles_m: int,
    tiles_n: int,
) -> torch.Tensor:
    pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank, dtype=torch.int32)
    pad = pad.view(-1, symm_mem_hdl.world_size)
    required = tiles_m * tiles_n
    if required > pad.size(0):
        msg = (
            "Signal pad has insufficient capacity for requested tiling: "
            f"need {required}, have {pad.size(0)}"
        )
        raise RuntimeError(msg)
    pad = pad[:required].contiguous()
    pad.zero_()
    return pad


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

    # Allocate shared symmetric memory buffers
    partial = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(partial, group=group)

    buf_tuple = tuple(
        symm_mem_hdl.get_buffer(rank, (M, N), torch.float32)
        for rank in range(symm_mem_hdl.world_size)
    )

    # Prepare signal pad for synchronization
    local_signal_pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank, dtype=torch.int32)

    # Calculate required signal pad size
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    num_tiles_m = helion.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = helion.cdiv(N, BLOCK_SIZE_N)
    num_blocks = num_tiles_m * num_tiles_n

    # Reshape signal pad and verify capacity
    local_signal_pad = local_signal_pad.view(-1, symm_mem_hdl.world_size)
    if num_blocks > local_signal_pad.size(0):
        # If signal pad is too small, we can either:
        # 1. Use a subset of tiles per launch
        # 2. Reuse signal pad locations with proper synchronization
        # For now, we'll error out to make the issue visible
        raise RuntimeError(
            f"Signal pad has insufficient capacity: need {num_blocks}, have {local_signal_pad.size(0)}. "
            f"This typically happens with large matrices. Consider using smaller tile sizes or "
            f"splitting the computation."
        )

    # Use only what we need and ensure it's zeroed
    local_signal_pad = local_signal_pad[:num_blocks].contiguous()
    local_signal_pad.zero_()

    signal_pad_addrs = _dev_array_to_tensor(
        symm_mem_hdl.signal_pad_ptrs_dev,
        (symm_mem_hdl.world_size,),
        torch.uint64,
    )

    output = torch.empty((M, N), dtype=torch.float32, device=a.device)

    return _gemm_one_shot_all_reduce_kernel(
        a,
        b,
        buf_tuple,
        local_signal_pad,
        signal_pad_addrs,
        output,
        BLOCK_SIZE_K=block_size_k,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
    )


def _reference_gemm_one_shot_all_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.matmul(a, b)
    dist.all_reduce(out)
    return out


def test(M: int, N: int, K: int, device: torch.device) -> None:
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    rank = dist.get_rank()
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)

    helion_result = helion_gemm_one_shot_all_reduce(a, b)
    reference = _reference_gemm_one_shot_all_reduce(a, b)

    torch.testing.assert_close(helion_result, reference, rtol=1e-1, atol=1e-1)


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(512, 256, 128, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --no_python python3 examples/gemm_one_shot_all_reduce.py
    """

    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
