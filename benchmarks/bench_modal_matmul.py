"""Benchmark 100 matmul configs on Modal H100s from a Mac (no local GPU needed).

Generates triton matmul kernels with different tile sizes, num_warps, and
num_stages, dispatches all 100 to Modal GPU workers, and reports the best config.

Usage:
    python benchmarks/bench_modal_matmul.py
"""

from __future__ import annotations

import itertools
import time

import torch

from helion.autotuner.modal_search import (
    ModalBenchmarkDispatcher,
    _find_compiled_fn_name,
    _serialize_args,
)

# Problem size: (M, K) x (K, N)
M, N, K = 4096, 4096, 4096
DTYPE = "float16"


def generate_matmul_triton_code(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    group_size_m: int = 8,
) -> str:
    """Generate a complete triton matmul kernel with the given config."""
    return f'''\
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k},
        GROUP_SIZE_M={group_size_m},
        num_warps={num_warps},
        num_stages={num_stages},
    )
    return c
'''


def generate_configs(n: int = 100) -> list[dict[str, int]]:
    """Generate n matmul configs by varying tile sizes, warps, and stages."""
    block_ms = [32, 64, 128, 256]
    block_ns = [32, 64, 128, 256]
    block_ks = [32, 64, 128]
    num_warps_list = [2, 4, 8]
    num_stages_list = [2, 3, 4, 5]

    all_combos = list(itertools.product(
        block_ms, block_ns, block_ks, num_warps_list, num_stages_list
    ))

    # Sample n configs evenly from the full grid
    step = max(1, len(all_combos) // n)
    selected = all_combos[::step][:n]

    # If we need more, pad with random sampling
    import random
    random.seed(42)
    while len(selected) < n and len(selected) < len(all_combos):
        candidate = random.choice(all_combos)
        if candidate not in selected:
            selected.append(candidate)

    configs = []
    for bm, bn, bk, nw, ns in selected:
        configs.append({
            "block_m": bm, "block_n": bn, "block_k": bk,
            "num_warps": nw, "num_stages": ns,
        })
    return configs[:n]


def main() -> None:
    configs = generate_configs(100)
    print(f"Matmul benchmark: {M}x{K} @ {K}x{N} ({DTYPE})")
    print(f"Generated {len(configs)} configs")
    print()

    # Create test matrices on CPU
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    args_bytes = _serialize_args([a, b])
    print(f"Serialized args: {len(args_bytes) / 1e6:.1f} MB")

    # Generate triton code for each config
    triton_codes: list[str] = []
    fn_names: list[str] = []
    valid_configs: list[dict[str, int]] = []

    for cfg in configs:
        code = generate_matmul_triton_code(**cfg)
        fn_name = _find_compiled_fn_name(code)
        triton_codes.append(code)
        fn_names.append(fn_name)
        valid_configs.append(cfg)

    print(f"Dispatching {len(triton_codes)} configs to Modal H100 workers...")
    print()

    # Dispatch all to Modal
    dispatcher = ModalBenchmarkDispatcher(gpu_type="H100", max_concurrent=100)
    t0 = time.time()

    try:
        results = dispatcher.dispatch_batch(
            triton_codes=triton_codes,
            fn_names=fn_names,
            args_bytes=args_bytes,
        )
    finally:
        dispatcher.close()

    elapsed = time.time() - t0

    # Collect and rank results
    ok_results: list[tuple[dict[str, int], float]] = []
    error_count = 0

    for cfg, result in zip(valid_configs, results, strict=True):
        status = result.get("status")
        perf = float(result.get("perf", float("inf")))
        if status == "ok" and perf < float("inf"):
            ok_results.append((cfg, perf))
        else:
            error_count += 1

    ok_results.sort(key=lambda x: x[1])

    print(f"Completed in {elapsed:.1f}s")
    print(f"  Successful: {len(ok_results)}/{len(configs)}")
    print(f"  Errors:     {error_count}/{len(configs)}")
    print()

    # Compute TFLOPS
    flops = 2.0 * M * N * K  # matmul FLOPs

    if ok_results:
        print("=" * 70)
        print("Top 10 configs:")
        print("=" * 70)
        print(f"{'Rank':<5} {'BLOCK_M':<9} {'BLOCK_N':<9} {'BLOCK_K':<9} "
              f"{'Warps':<7} {'Stages':<8} {'Time (ms)':<11} {'TFLOPS':<8}")
        print("-" * 70)
        for i, (cfg, perf) in enumerate(ok_results[:10]):
            tflops = flops / (perf * 1e-3) / 1e12
            print(f"{i+1:<5} {cfg['block_m']:<9} {cfg['block_n']:<9} "
                  f"{cfg['block_k']:<9} {cfg['num_warps']:<7} "
                  f"{cfg['num_stages']:<8} {perf:<11.4f} {tflops:<8.1f}")

        print()
        print("=" * 70)
        best_cfg, best_perf = ok_results[0]
        best_tflops = flops / (best_perf * 1e-3) / 1e12
        print(f"BEST CONFIG: {best_perf:.4f} ms ({best_tflops:.1f} TFLOPS)")
        print(f"  BLOCK_M     = {best_cfg['block_m']}")
        print(f"  BLOCK_N     = {best_cfg['block_n']}")
        print(f"  BLOCK_K     = {best_cfg['block_k']}")
        print(f"  num_warps   = {best_cfg['num_warps']}")
        print(f"  num_stages  = {best_cfg['num_stages']}")
        print("=" * 70)
    else:
        print("No configs succeeded!")
        # Show some errors for debugging
        for cfg, result in zip(valid_configs[:3], results[:3]):
            print(f"  {cfg}: {result.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
