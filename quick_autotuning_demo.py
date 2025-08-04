#!/usr/bin/env python3

import logging
import helion
import helion.language as hl
import torch
from triton.testing import do_bench

logging.getLogger().setLevel(logging.WARNING)

print("Quick Helion Configuration Comparison")
print("=" * 60)

# Simple kernel: Fused multiply-add with ReLU
def fma_relu_spec(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.relu(a * b + c)

# Version 1: Small block size
@helion.kernel(config=helion.Config(block_sizes=[64]))
def fma_relu_small_blocks(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.shape):
        out[tile] = torch.relu(a[tile] * b[tile] + c[tile])
    return out

# Version 2: Large block size
@helion.kernel(config=helion.Config(block_sizes=[1024]))
def fma_relu_large_blocks(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.shape):
        out[tile] = torch.relu(a[tile] * b[tile] + c[tile])
    return out

# Version 3: Default config (use_default_config=True for quick iteration)
@helion.kernel(use_default_config=True)
def fma_relu_default(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.shape):
        out[tile] = torch.relu(a[tile] * b[tile] + c[tile])
    return out

# Test with a large 1D tensor
size = 10_000_000
a = torch.randn(size, device="cuda", dtype=torch.float32)
b = torch.randn(size, device="cuda", dtype=torch.float32)
c = torch.randn(size, device="cuda", dtype=torch.float32)

print(f"Testing on tensor of size {size:,}")
print("-" * 40)

# Verify correctness
expected = fma_relu_spec(a, b, c)
for name, kernel in [
    ("Small blocks (64)", fma_relu_small_blocks),
    ("Large blocks (1024)", fma_relu_large_blocks),
    ("Default config", fma_relu_default),
]:
    result = kernel(a, b, c)
    torch.testing.assert_close(result, expected)
    print(f"✅ {name} correct")

# Benchmark
print("\nPerformance comparison:")
print("-" * 40)

pytorch_time = do_bench(lambda: fma_relu_spec(a, b, c))
print(f"PyTorch baseline: {pytorch_time:.3f} ms")

for name, kernel in [
    ("Small blocks (64)", fma_relu_small_blocks),
    ("Large blocks (1024)", fma_relu_large_blocks),
    ("Default config", fma_relu_default),
]:
    kernel_time = do_bench(lambda: kernel(a, b, c))
    speedup = pytorch_time / kernel_time
    print(f"{name}: {kernel_time:.3f} ms (speedup: {speedup:.2f}x)")

print("\n" + "=" * 60)
print("Key insights:")
print("1. Block size significantly affects performance")
print("2. Optimal block size depends on hardware and problem size")
print("3. use_default_config=True provides reasonable performance for development")
print("4. For production, remove config to enable full autotuning")
print("=" * 60)