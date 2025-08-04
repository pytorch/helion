#!/usr/bin/env python3

import logging
from typing import Tuple
import helion
import helion.language as hl
import torch
from torch import Tensor
from triton.testing import do_bench

logging.getLogger().setLevel(logging.WARNING)

def test_kernel(kernel_fn, spec_fn, *args, **kwargs):
    """Test a Helion kernel against a reference implementation."""
    result = kernel_fn(*args, **kwargs)
    expected = spec_fn(*args, **kwargs)
    torch.testing.assert_close(result, expected)
    print("✅ Results Match ✅")

def benchmark_kernel(kernel_fn, *args, **kwargs):
    """Benchmark a Helion kernel."""
    no_args = lambda: kernel_fn(*args, **kwargs)
    time_in_ms = do_bench(no_args)
    return time_in_ms

def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = do_bench(kernel_no_args)
    spec_time = do_bench(spec_no_args)
    print(f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time/kernel_time:.3f}x")

# Example 1: Constant Add
def add_spec(x: Tensor) -> Tensor:
    """This is the spec that you should implement."""
    return x + 10.0

@helion.kernel(config=helion.Config(block_sizes=[256]))
def add_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    TILE_RANGE = x.shape
    for tile_n in hl.tile(TILE_RANGE):
        out[tile_n] = x[tile_n] + 10.0
    return out

# Example 2: Outer Vector Add
def broadcast_add_spec(x: Tensor, y: Tensor) -> Tensor:
    return x[None, :] + y[:, None]

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def broadcast_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    for tile_i, tile_j in hl.tile([n1, n0]):
        out[tile_i, tile_j] = x[None, tile_j] + y[tile_i, None]
    return out

# Example 3: Fused Outer Multiplication
def mul_relu_block_spec(x: Tensor, y: Tensor) -> Tensor:
    return torch.relu(x[None, :] * y[:, None])

@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def mul_relu_block_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n0 = x.size(0)
    n1 = y.size(0)
    out = torch.empty([n1, n0], dtype=x.dtype, device=x.device)
    for tile_i, tile_j in hl.tile([n1, n0]):
        out[tile_i, tile_j] = torch.relu(x[None, tile_j] * y[tile_i, None])
    return out

# Example 5: Long Sum
def sum_spec(x: torch.Tensor) -> torch.Tensor:
    return x.sum(1)

@helion.kernel(config=helion.Config(block_sizes=[1, 16]))
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, hidden = x.size()
    out = torch.empty([batch, hidden], dtype=x.dtype, device=x.device)
    for tile_batch, tile_hidden in hl.tile([batch, hidden]):
        out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
    return out

# Example 6: Jagged Tensor Addition
def jagged_dense_add_spec(x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference implementation in pure PyTorch."""
    num_rows = x_offsets.numel() - 1
    assert y.shape[0] == num_rows
    out = y.clone()
    for i in range(num_rows):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        out[i, 0 : end - start] += torch.mean(x_data[start:end])
    return out

@helion.kernel(config=helion.Config(block_sizes=[1, 512, 512, 512], num_warps=8, num_stages=6))
def jagged_dense_add_kernel(x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add the mean of a jagged sparse tensor to a dense matrix."""
    num_rows = y.size(0)
    out = torch.zeros_like(y)
    
    for tile0 in hl.tile(num_rows):
        starts = x_offsets[tile0]
        ends = x_offsets[tile0.index + 1]
        nnz = ends - starts
        max_nnz = nnz.amax()
        
        row_sum = hl.zeros([tile0], dtype=x_data.dtype)
        
        for tile1 in hl.tile(0, max_nnz):
            x_slice = hl.load(
                x_data,
                [starts[:, None] + tile1.index[None, :]],
                extra_mask=tile1.index[None, :] < nnz[:, None]
            )
            row_sum += x_slice.sum(axis=1)
        
        row_mean = torch.where(nnz > 0, row_sum / nnz.float(), torch.zeros_like(row_sum))
        
        for tile1 in hl.tile(0, max_nnz):
            mask = tile1.index[None, :] < nnz[:, None]
            out[tile0, tile1] = torch.where(
                mask, 
                y[tile0, tile1] + row_mean[:, None], 
                y[tile0, tile1]
            )
        
        for tile1 in hl.tile(max_nnz, y.size(1)):
            out[tile0, tile1] = y[tile0, tile1]

    return out

def random_jagged_2d(num_rows: int, max_cols: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random jagged tensor data for testing."""
    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    x_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)])
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, dtype=dtype, device=device)
    return x_data, x_offsets

# Test all examples
def main():
    print("=" * 60)
    print("Testing Example 1: Constant Add")
    x = torch.randn(8192, device="cuda")
    test_kernel(add_kernel, add_spec, x)
    benchmark_kernel(add_kernel, x)
    compare_implementations(add_kernel, add_spec, x)
    
    print("\n" + "=" * 60)
    print("Testing Example 2: Outer Vector Add")
    x = torch.randn(1142, device="cuda")
    y = torch.randn(512, device="cuda")
    test_kernel(broadcast_add_kernel, broadcast_add_spec, x, y)
    benchmark_kernel(broadcast_add_kernel, x, y)
    compare_implementations(broadcast_add_kernel, broadcast_add_spec, x, y)
    
    print("\n" + "=" * 60)
    print("Testing Example 3: Fused Outer Multiplication")
    x = torch.randn(512, device="cuda")
    y = torch.randn(512, device="cuda")
    test_kernel(mul_relu_block_kernel, mul_relu_block_spec, x, y)
    compare_implementations(mul_relu_block_kernel, mul_relu_block_spec, x, y)
    
    print("\n" + "=" * 60)
    print("Testing Example 5: Long Sum")
    x = torch.randn(128, 129, 128, device="cuda")
    test_kernel(sum_kernel, sum_spec, x)
    compare_implementations(sum_kernel, sum_spec, x)
    
    print("\n" + "=" * 60)
    print("Testing Example 6: Jagged Tensor Addition")
    rows, cols = 64, 1024
    x_data, x_offsets = random_jagged_2d(rows, cols, device="cuda")
    y = torch.randn(rows, cols, device="cuda")
    test_kernel(jagged_dense_add_kernel, jagged_dense_add_spec, x_data, x_offsets, y)
    compare_implementations(jagged_dense_add_kernel, jagged_dense_add_spec, x_data, x_offsets, y)
    
    print("\n" + "=" * 60)
    print("All tests passed! 🎉")

if __name__ == "__main__":
    main()