#!/usr/bin/env python3

import logging
import helion
import helion.language as hl
import torch
from triton.testing import do_bench
import time

logging.getLogger().setLevel(logging.WARNING)

print("Helion Autotuning Comparison")
print("=" * 60)
print("This demonstrates the performance difference between manually")
print("configured kernels and autotuned kernels.")
print("=" * 60)

# Example: Matrix multiplication followed by ReLU
def matmul_relu_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.matmul(a, b))

# Manually configured kernel
@helion.kernel(config=helion.Config(block_sizes=[32, 32, 32], num_warps=4))
def matmul_relu_manual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=a.dtype)
        for tile_k in hl.tile(k):
            acc += a[tile_m, tile_k] @ b[tile_k, tile_n]
        out[tile_m, tile_n] = torch.relu(acc)
    
    return out

# Autotuned kernel (no config specified)
@helion.kernel()
def matmul_relu_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=a.dtype)
        for tile_k in hl.tile(k):
            acc += a[tile_m, tile_k] @ b[tile_k, tile_n]
        out[tile_m, tile_n] = torch.relu(acc)
    
    return out

# Test different matrix sizes
test_sizes = [
    (512, 512, 512),
    (1024, 768, 768),
    (2048, 1024, 1024),
]

for m, k, n in test_sizes:
    print(f"\nMatrix size: ({m}, {k}) x ({k}, {n})")
    print("-" * 40)
    
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    
    # Verify correctness
    manual_result = matmul_relu_manual(a, b)
    expected = matmul_relu_spec(a, b)
    torch.testing.assert_close(manual_result, expected, rtol=1e-3, atol=1e-3)
    print("✅ Manual kernel correct")
    
    # Benchmark manual config
    manual_time = do_bench(lambda: matmul_relu_manual(a, b))
    pytorch_time = do_bench(lambda: matmul_relu_spec(a, b))
    
    print(f"Manual config time: {manual_time:.3f} ms")
    print(f"PyTorch time: {pytorch_time:.3f} ms")
    print(f"Manual speedup: {pytorch_time/manual_time:.2f}x")
    
    # Note: Autotuning happens on first call and can take significant time
    print("\n🔧 Running autotuning (this may take a while on first run)...")
    start_time = time.time()
    
    # First call triggers autotuning
    autotuned_result = matmul_relu_autotuned(a, b)
    torch.testing.assert_close(autotuned_result, expected, rtol=1e-3, atol=1e-3)
    
    autotune_duration = time.time() - start_time
    print(f"✅ Autotuned kernel correct (tuning took {autotune_duration:.1f}s)")
    
    # Benchmark autotuned version (subsequent calls use cached config)
    autotuned_time = do_bench(lambda: matmul_relu_autotuned(a, b))
    
    print(f"Autotuned time: {autotuned_time:.3f} ms")
    print(f"Autotuned speedup: {pytorch_time/autotuned_time:.2f}x")
    print(f"Autotuning improvement: {manual_time/autotuned_time:.2f}x over manual config")

print("\n" + "=" * 60)
print("Conclusion: Autotuning can significantly improve performance")
print("by finding optimal configurations for your specific hardware")
print("and problem size. The tuning cost is one-time per configuration.")
print("=" * 60)