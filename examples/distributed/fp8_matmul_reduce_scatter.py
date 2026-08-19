"""
FP8 MatMul + Reduce-Scatter Fusion Example
==========================================
This example extends the matmul_reduce_scatter example to use FP8 inputs.
Each rank holds FP8 A and B shards; the kernel computes a local FP8 GEMM
(accumulating in FP32 via ``hl.dot``), applies per-row/per-column scales,
writes the bfloat16 partial result to a symmetric-memory buffer, performs an
intra-group barrier, and then reduce-scatters: each rank accumulates the rows
it owns from all peers' buffers, producing a ``[M//WORLD_SIZE, N]`` bfloat16
output.
"""

from __future__ import annotations

import functools
import os

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion._testing import DEVICE
from helion._testing import assert_close_with_mismatch_tolerance
from helion._testing import run_example
import helion.language as hl
from helion.runtime.dist_utils import symm_mem_sync

tolerance = {
    "atol": 5e-1,
    "rtol": 5e-1,
    "max_mismatch_pct": 1e-3,
}


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64, 32],  # M, N, K
        num_warps=8,
        num_stages=3,
    ),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    autotune_baseline_accuracy_check_fn=functools.partial(
        assert_close_with_mismatch_tolerance,
        **tolerance,
    ),
)
def fp8_matmul_reduce_scatter_kernel(
    a: torch.Tensor,  # [M, K] float8_e4m3fn
    b: torch.Tensor,  # [K, N] float8_e4m3fn
    scale_a: torch.Tensor,  # [M, 1] float32
    scale_b: torch.Tensor,  # [1, N] float32
    symm_mem_buffer: torch.Tensor,  # [M, N] bfloat16, symmetric memory
    signal_pad_ptrs: torch.Tensor,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    GROUP_NAME: hl.ProcessGroupName,
) -> torch.Tensor:
    """
    Fused FP8 MatMul + Reduce-Scatter kernel.

    Computes ``(scale_a * scale_b * (A @ B)).to(bfloat16)`` in a distributed
    reduce-scatter pattern: each rank emits only its ``M // WORLD_SIZE`` output rows.
    """
    M, K = a.size()
    K2, N = b.size()
    M_scatter = M // WORLD_SIZE  # type: ignore[unsupported-operation]

    output = torch.empty([M_scatter, N], dtype=torch.bfloat16, device=a.device)

    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, GROUP_NAME)

    scatter_start = RANK * M_scatter  # type: ignore[unsupported-operation]
    scatter_end = scatter_start + M_scatter  # type: ignore[unsupported-operation]

    for tile_m, tile_n in hl.tile([M, N]):
        # FP8 GEMM tile, accumulating in FP32
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)

        # Apply per-row and per-column scales
        acc = (
            acc
            * scale_a[tile_m, :].to(torch.float32)
            * scale_b[:, tile_n].to(torch.float32)
        )

        # Store bfloat16 partial result to this rank's symmetric-memory buffer
        symm_mem_buffer[tile_m, tile_n] = acc.to(torch.bfloat16)

        # Barrier: release our write, acquire peers' writes
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, None, RANK, WORLD_SIZE, True, True),
            output_like=None,
        )

        # Reduce-scatter: accumulate only the rows this rank owns
        if tile_m.begin >= scatter_start and tile_m.begin < scatter_end:  # type: ignore[unsupported-operation]
            acc_reduce = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for remote_buffer in buffer_tuple:
                acc_reduce = acc_reduce + remote_buffer[tile_m, tile_n].to(
                    torch.float32
                )
            output[tile_m.index - scatter_start, tile_n] = acc_reduce.to(torch.bfloat16)  # type: ignore[unsupported-operation]

        # Final barrier (release only)
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, None, RANK, WORLD_SIZE, True, False),
            output_like=None,
        )

    return output


def helion_fp8_matmul_reduce_scatter(
    symm_mem_buffer: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper that rendezvouss on the pre-allocated symmetric buffer and
    invokes the FP8 reduce-scatter kernel.

    Args:
        symm_mem_buffer: Pre-allocated symmetric-memory buffer ``[M, N]`` bfloat16.
        a: Local FP8 A shard ``[M, K]`` (``torch.float8_e4m3fn``).
        b: Local FP8 B shard ``[K, N]`` (``torch.float8_e4m3fn``).
        scale_a: Per-row scale ``[M, 1]`` float32.
        scale_b: Per-column scale ``[1, N]`` float32.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group.group_name)

    return fp8_matmul_reduce_scatter_kernel(
        a,
        b,
        scale_a,
        scale_b,
        symm_mem_buffer,
        symm_mem_hdl.signal_pad_ptrs_dev,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        GROUP_NAME=group.group_name,
    )


def reference_fp8_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Reference: FP8 scaled matmul, reduce-scatter along M.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    c = torch._scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
    )

    world_size = dist.get_world_size(group)
    M_scatter = c.shape[0] // world_size
    output = torch.empty(M_scatter, c.shape[1], dtype=c.dtype, device=c.device)
    dist.reduce_scatter_tensor(output, c, group=group)
    return output


def reference_fused_scaled_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Reference using PyTorch's built-in
    ``_fused_scaled_matmul_reduce_scatter`` kernel.
    """
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    M, N = a.shape[0], b.shape[1]
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        a,
        b,
        scale_a,
        scale_b,
        reduce_op="sum",
        orig_scatter_dim=0,
        scatter_dim_after_maybe_reshape=0,
        group_name=group.group_name,
        output_shape=[M, N],
        out_dtype=torch.bfloat16,
    )


def test(M: int, N: int, K: int, device: torch.device) -> None:
    """Test the FP8 reduce-scatter kernel against the reference."""
    rank = dist.get_rank()

    torch.manual_seed(23 + rank)
    a_fp32 = torch.randn(M, K, device=device)
    a = a_fp32.to(torch.float8_e4m3fn)

    torch.manual_seed(23)
    b_fp32 = torch.randn(K, N, device=device)
    b = b_fp32.to(torch.float8_e4m3fn).t().contiguous().t()

    scale_a = torch.rand(M, 1, device=device)
    scale_b = torch.rand(1, N, device=device)

    symm_mem_buffer = symm_mem.empty(M, N, dtype=torch.bfloat16, device=device)
    symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD.group_name)  # type: ignore[union-attr]

    run_example(
        functools.partial(helion_fp8_matmul_reduce_scatter, symm_mem_buffer),
        {
            "nccl+cublas": reference_fp8_matmul_reduce_scatter,
            "fused_baseline": reference_fused_scaled_matmul_reduce_scatter,
        },
        (a, b, scale_a, scale_b),
        **tolerance,
    )


def main() -> None:
    _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    test(M=512, N=768, K=1024, device=device)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \\
    --nproc-per-node 4 \\
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \\
    examples/distributed/fp8_matmul_reduce_scatter.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
