# Issue #467: Error when autotuning kernel

## Metadata
- **State**: OPEN
- **Author**: [angelayi](https://github.com/angelayi)
- **Created**: August 08, 2025 at 04:51 UTC
- **Updated**: August 08, 2025 at 04:51 UTC

## Description

The following script script runs successfully with `HELION_USE_DEFAULT_CONFIG=1` but fails when I remove it. 

```python
from __future__ import annotations

import torch

import triton
import triton
import triton.language as tl

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel
def bf16xint16_gemm_fwd(x: torch.Tensor, w: torch.Tensor, transpose: hl.constexpr=False):
    m, k = x.shape
    k1, n = w.shape
    assert k == k1, f"size mismatch: {k} != {k1}"
    assert x.is_contiguous(), "x must be contiguous"

    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulate in FP32 for accuracy
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            x_tile = x[tile_m, tile_k]
            w_tile = w[tile_k, tile_n].to(torch.bfloat16)
            acc = hl.dot(x_tile, w_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def bf16xint16_triton_matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    TRANSPOSE: tl.constexpr,  # if true, assume a_ptr is int16; otherwise assume b_ptr is int16
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        if TRANSPOSE:
            tl.static_assert(a.dtype == tl.int16)
            tl.static_assert(b.dtype == tl.bfloat16)
            a_bf16 = a.to(tl.bfloat16)
            b_bf16 = b
        else:
            tl.static_assert(a.dtype == tl.bfloat16)
            tl.static_assert(b.dtype == tl.int16)
            a_bf16 = a
            b_bf16 = b.to(tl.bfloat16)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a_bf16, b_bf16, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bf16xint16_triton_matmul(a, b, transpose=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    bf16xint16_triton_matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        TRANSPOSE=transpose,
    )
    return c


def main() -> None:
    batch_size = 32
    dout = 1280
    din = 8192
    device = "cuda"

    x = torch.randn([batch_size, din], device=device, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15, [din, dout], device=device, dtype=torch.int16)

    run_example(
        bf16xint16_gemm_fwd,
        bf16xint16_triton_matmul,
        (x, w),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    main()
```

Full paste of the error message is in [P1896285623](https://www.internalfb.com/phabricator/paste/view/P1896285623). Jason mentioned it had something to do with not being able to find the right block size? 
```
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
Process ForkProcess-2:
Traceback (most recent call last):
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/core.py", line 2045, in dot
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/semantic.py", line 1504, in dot
    and rhs.shape[-1].value >= min_dot_size[1], \
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/data/users/angelayi/helion/helion/runtime/precompile_shim.py", line 54, in finish_it
    kernel_cache[key] = fn.compile(
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/compiler/compiler.py", line 339, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/compiler/compiler.py", line 83, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
triton.compiler.errors.CompilationError: at 23:14:
    pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
    pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
    offset_1 = pid_0 * _BLOCK_SIZE_1
    offset_0 = pid_1
    acc = tl.full([1, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in tl.range(0, k.to(tl.int32), _BLOCK_SIZE_2, loop_unroll_factor=1, num_stages=3):
        acc_copy = acc
        acc_copy_0 = acc_copy
        x_tile = x_desc.load([offset_0, offset_2])
        load_1 = w_desc.load([offset_2, offset_1])
        v_0 = load_1.to(tl.bfloat16)
        acc = tl.dot(x_tile, v_0, acc=acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
              ^
Process ForkProcess-3:
Traceback (most recent call last):
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/core.py", line 2045, in dot
    return _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)
  File "/home/angelayi/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/triton/language/semantic.py", line 1504, in dot
    and rhs.shape[-1].value >= min_dot_size[1], \
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16
```


## Comments

*No comments yet.*
