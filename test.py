from __future__ import annotations

import torch
from triton.testing import do_bench

import helion
import helion.language as hl

# Set seed for reproducibility
torch.manual_seed(42)


@helion.kernel(autotune_random_seed=42)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out


x = torch.randn([2048, 2048], device="cuda")
y = torch.randn([2048, 2048], device="cuda")

# Run once to trigger autotuning
print("Running autotuning...")
out = matmul(x, y)

# Benchmark the final kernel
print("\nBenchmarking optimized kernel...")
time_ms = do_bench(lambda: matmul(x, y), return_mode="median")
print(f"Median time: {time_ms:.4f} ms")

# Calculate TFLOPS
m, k = x.shape
n = y.shape[1]
flops = 2 * m * n * k  # 2 * M * N * K for matmul
tflops = flops / (time_ms * 1e-3) / 1e12
print(f"Performance: {tflops:.2f} TFLOPS")

print(f"\nOutput shape: {out.shape}")
