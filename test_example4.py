#!/usr/bin/env python3

import logging
import helion
import helion.language as hl
import torch
from triton.testing import do_bench

logging.getLogger().setLevel(logging.WARNING)

# Example 4: Using next_power_of_2() for Efficient Tiling
def chunked_sum_spec(x: torch.Tensor) -> torch.Tensor:
    """Compute the sum of elements in a 2D tensor along the last dimension."""
    return x.sum(-1)

@helion.kernel(config=helion.Config(block_sizes=[1, 128]))
def chunked_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Sum using power-of-2 chunk processing for efficiency."""
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    
    # Calculate an efficient chunk size based on the input dimension
    # This demonstrates using next_power_of_2 for optimization hints
    chunk_size = helion.next_power_of_2(min(n, 128))
    
    # Register the block size so it's available in the device loop
    block_size_n = hl.register_block_size(chunk_size)
    
    for tile_m in hl.tile(m):
        # Initialize accumulator with the registered block size
        acc = hl.zeros([tile_m, block_size_n], dtype=x.dtype)
        
        # Process in chunks of power-of-2 size
        for tile_n in hl.tile(n, block_size=block_size_n):
            acc += x[tile_m, tile_n]
        
        # Reduce the accumulator
        out[tile_m] = acc.sum(-1)
    
    return out

def test_kernel(kernel_fn, spec_fn, *args, **kwargs):
    """Test a Helion kernel against a reference implementation."""
    result = kernel_fn(*args, **kwargs)
    expected = spec_fn(*args, **kwargs)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("✅ Results Match ✅")

def compare_implementations(kernel_fn, spec_fn, *args, **kwargs):
    """Benchmark a Helion kernel and its reference implementation."""
    kernel_no_args = lambda: kernel_fn(*args, **kwargs)
    spec_no_args = lambda: spec_fn(*args, **kwargs)
    kernel_time = do_bench(kernel_no_args)
    spec_time = do_bench(spec_no_args)
    print(f"⏱ Helion Kernel Time: {kernel_time:.3f} ms, PyTorch Reference Time: {spec_time:.3f} ms, Speedup: {spec_time/kernel_time:.3f}x")

# Test the kernel
print("Testing Example 4: next_power_of_2 usage")
x = torch.randn(256, 1234, device="cuda")  # Non-power-of-2 width
test_kernel(chunked_sum_kernel, chunked_sum_spec, x)
compare_implementations(chunked_sum_kernel, chunked_sum_spec, x)
print("Example 4 test passed! 🎉")