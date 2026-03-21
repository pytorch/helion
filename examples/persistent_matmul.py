"""
Persistent Matmul Kernel Example
=================================
This example demonstrates a persistent matrix multiplication kernel in Helion,
inspired by the Triton persistent matmul tutorial. Instead of mapping each thread
block to one output tile, this kernel launches one thread block per streaming
multiprocessor (SM) and has each SM process multiple output tiles in a loop.
This approach improves GPU occupancy and load balancing.

The key difference from the standard matmul (``examples/matmul.py``) is the work
distribution strategy:

- **Standard matmul**: Each thread block computes exactly one output tile. The total
  number of thread blocks equals the number of output tiles.

- **Persistent matmul**: The number of thread blocks equals the number of SMs. Each
  thread block loops over multiple output tiles using a strided assignment pattern
  for load balancing.

This file includes both the Helion implementation and the original Triton tutorial
kernel for side-by-side comparison.

Reference: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
"""

# ruff: noqa: ANN001, ANN201, ANN202
# Triton JIT functions use tl.constexpr, not Python type annotations.

# %%
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# Triton tutorial uses: BLOCK_M=128, BLOCK_N={128,256}, BLOCK_K={64,128},
# num_warps={4,8}, num_stages={2,3,4}, GROUP_SIZE_M=8.
# We pick one representative config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
# num_warps=8 to match the tutorial's autotuning search space.
config = helion.Config(
    block_sizes=[128, 128, 64],
    num_warps=8,
    range_flattens=[None, True, None],
)


# %%
# Helion Implementation
# ---------------------


# %%
@helion.kernel(static_shapes=False, config=config)
def persistent_matmul(
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """
    Persistent matrix multiplication kernel.

    Each SM is assigned as a persistent worker that loops over multiple output
    tiles using a strided distribution pattern for load balancing.

    Args:
        x: Left input matrix of shape [M, K].
        y: Right input matrix of shape [K, N].

    Returns:
        Output matrix of shape [M, N].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"

    device = x.device
    if device.type == "xpu":
        num_workers = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    else:
        num_workers = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count

    # Define tunable block sizes for M, N dimensions (auto-tuned at runtime)
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)

    out = torch.zeros(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=device
    )

    # Compute tile grid dimensions
    num_m_tiles = (m + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_m_tiles * num_n_tiles

    # Tile swizzle group size for L2 cache locality (matches Triton tutorial)
    GROUP_SIZE_M = 8
    num_pid_in_group = GROUP_SIZE_M * num_n_tiles

    for worker_id in hl.grid(num_workers):
        # Persistent thread pattern: each worker processes tiles across
        # the output using strided/interleaved assignment for load balancing.
        # Uses strided range: worker 0 gets tiles 0, N, 2N, ...;
        # worker 1 gets tiles 1, N+1, 2N+1, ...; etc.
        for tile_idx in hl.grid(worker_id, num_tiles, step=num_workers):
            # Swizzle tile index for L2 cache locality (GROUP_SIZE_M grouping)
            # pyrefly: ignore[unsupported-operation]
            group_id = tile_idx // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            m_tile_idx = first_pid_m + ((tile_idx % num_pid_in_group) % group_size_m)
            n_tile_idx = (tile_idx % num_pid_in_group) // group_size_m

            # Compute global memory offsets for this tile
            base_row = m_tile_idx * BLOCK_M
            # pyrefly: ignore[unsupported-operation]
            base_col = n_tile_idx * BLOCK_N

            # Generate row and column index ranges for tile access
            row_idx = base_row + hl.arange(BLOCK_M)
            col_idx = base_col + hl.arange(BLOCK_N)

            # Clamp OOB indices to 0 (matching Triton tutorial pattern)
            # This allows removing M/N masks from loads since all indices
            # point to valid memory (row/col 0 is always valid).
            rows_valid = row_idx < m
            cols_valid = col_idx < n
            row_idx = torch.where(rows_valid, row_idx, 0)
            col_idx = torch.where(cols_valid, col_idx, 0)

            # Contiguity/alignment hints for better memory coalescing
            row_idx = hl.inline_triton(
                "tl.max_contiguous(tl.multiple_of({0}, {1}), {1})",
                args=[row_idx, BLOCK_M],
                output_like=row_idx,
            )
            col_idx = hl.inline_triton(
                "tl.max_contiguous(tl.multiple_of({0}, {1}), {1})",
                args=[col_idx, BLOCK_N],
                output_like=col_idx,
            )

            # Initialize FP32 accumulator for numerical precision
            acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)

            # Iterate over K dimension in blocks for matrix multiplication
            for k_tile in hl.tile(k):
                k_idx = k_tile.index

                # Load tiles — no M/N mask needed (OOB clamped to 0)
                a_blk = hl.load(x, [row_idx, k_idx])
                b_blk = hl.load(y, [k_idx, col_idx])

                # Perform tile-level matrix multiplication and accumulate
                acc = torch.addmm(acc, a_blk, b_blk)

            # Store still needs mask — must not write to wrong locations
            # pyrefly: ignore[bad-index]
            valid_2d = rows_valid[:, None] & cols_valid[None, :]
            hl.store(
                out,
                [row_idx, col_idx],
                acc.to(out.dtype),
                extra_mask=valid_2d,
            )

    return out


# %%
# Triton Tutorial Kernel (for comparison)
# ----------------------------------------
# The following is the persistent matmul kernel from the Triton tutorial,
# included for side-by-side comparison with the Helion implementation above.
# Note how Helion abstracts away manual pointer arithmetic, mask computation,
# index clamping, and grid launch configuration.


# %%
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def triton_persistent_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it
    # can't handle a value being used in both the prologue and epilogue, so
    # we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs,
                mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def triton_persistent_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Wrapper that launches the Triton persistent matmul kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (min(NUM_SMS, triton.cdiv(M, 128) * triton.cdiv(N, 128)),)
    triton_persistent_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
    )
    return c


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Check correctness of both kernels against PyTorch's matmul.

    Args:
        m: Number of rows in the left input matrix.
        k: Shared dimension.
        n: Number of columns in the right input matrix.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
    print("--- Helion persistent matmul ---")
    run_example(persistent_matmul, torch.matmul, (x, y), rtol=1e-2, atol=1e-2)
    print("\n--- Triton tutorial persistent matmul ---")
    run_example(triton_persistent_matmul, torch.matmul, (x, y), rtol=1e-2, atol=1e-2)


# %%
def main() -> None:
    check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
