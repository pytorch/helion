"""
Chunkwise Linear Attention: Model Comparison

Three different linear attention models -- all using the same underlying
Helion kernel, parameterized differently.
"""

from __future__ import annotations

import time

import torch
from chunk_recurrence import (
    gated_delta_net_fwd_h,
    linear_attention_fwd_h,
    delta_net_fwd_h,
)


def benchmark_variant(name, kernel_fn, k, w, u, g, warmup=10, repeat=100):
    for _ in range(warmup):
        kernel_fn(k, w, u, g)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        kernel_fn(k, w, u, g)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {name}: {median:.1f} us")
    return median


def main():
    device = "cuda"
    B, T, H, K, V = 2, 1024, 4, 64, 64

    print(f"B={B}, T={T}, H={H}, K={K}, V={V}, chunks={T // 64}")

    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    u = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.cumsum(
        -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device)),
        dim=1,
    )

    gdn = gated_delta_net_fwd_h(64)
    lin = linear_attention_fwd_h(64)
    dlt = delta_net_fwd_h(64)

    print("Same primitive, different update rules:")
    benchmark_variant("Gated DeltaNet", gdn, k, w, u, g)
    benchmark_variant("Linear/RetNet ", lin, k, w, u, g)
    benchmark_variant("DeltaNet      ", dlt, k, w, u, g)


if __name__ == "__main__":
    main()
