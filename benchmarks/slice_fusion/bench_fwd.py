"""
Forward kernel benchmark for matmul + slice fusion (LCE pattern).

Shape: (B=4096, M=128, N=192, K=48, slice_size=96)
transpose_A=True, transpose_C=True, 1D bias.

Forward: output[B,N,M] = (A[B,K,M]^T @ B_w[N,K]^T + bias[N,None]).contiguous()
Slice:   sliced = output[:, :slice_size, :]

Compares:
  1. Inductor (torch.compile) — uses cuBLAS + pointwise kernels; slice is a view
  2. Helion-Original: fused matmul+bias+store, with conditional slice store
  3. Helion-View: fused matmul+bias+store, slice as view outside kernel

Context: https://github.com/pytorch/helion/issues/1679

Usage:
  CUDA_VISIBLE_DEVICES=7 triton/scripts/denoise.sh python benchmarks/slice_fusion/bench_fwd.py
"""
from __future__ import annotations

import os
import torch
import helion
import helion.language as hl

B, M, N, K, SLICE = 4096, 128, 192, 48, 96
WARMUP, ITERS = 20, 100


# ═════════════════════════════════════════════════════════════════════════
# 1. Inductor baseline
# ═════════════════════════════════════════════════════════════════════════
@torch.compile
def inductor_fwd(A, B_w, bias, slice_size):
    A = A.to(torch.bfloat16)
    B_w = B_w.to(torch.bfloat16)
    bias = bias.to(torch.bfloat16)
    out = torch.matmul(A.transpose(-2, -1), B_w.T)
    out = out.transpose(-2, -1).contiguous()
    out = (out + bias.unsqueeze(-1)).contiguous()
    return out, out[:, :slice_size, :]


# ═════════════════════════════════════════════════════════════════════════
# 2. Helion-Original: conditional tile store for slice
# ═════════════════════════════════════════════════════════════════════════
@helion.kernel()
def helion_fwd_orig(
    A: torch.Tensor,       # [B, K, M]
    B_w: torch.Tensor,     # [N, K]
    bias: torch.Tensor,    # [N]
    slice_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    Bat = A.size(0)
    Kd = A.size(1)
    Md = hl.specialize(A.size(2))
    Nd = hl.specialize(B_w.size(0))

    output = torch.empty((Bat, Nd, Md), device=A.device, dtype=torch.bfloat16)
    sliced = torch.empty((Bat, slice_size, Md), device=A.device, dtype=torch.bfloat16)

    for tb in hl.tile(Bat, block_size=1):
        for tm, tn in hl.tile([Md, Nd]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(Kd):
                a = A[tb.begin, tk, tm].to(torch.bfloat16)
                b = B_w[tn, tk].to(torch.bfloat16)
                acc = torch.addmm(acc, a.t(), b.t())

            acc = acc + bias[tn].to(torch.bfloat16)[None, :]
            r = acc.to(torch.bfloat16).t()

            output[tb.begin, tn, tm] = r
            if tn.end <= slice_size:
                sliced[tb.begin, tn, tm] = r

    return output, sliced


# ═════════════════════════════════════════════════════════════════════════
# 3. Helion-View: compute full output, slice as view outside kernel
# ═════════════════════════════════════════════════════════════════════════
@helion.kernel()
def helion_fwd_view_k(
    A: torch.Tensor,       # [B, K, M]
    B_w: torch.Tensor,     # [N, K]
    bias: torch.Tensor,    # [N]
) -> torch.Tensor:
    Bat = A.size(0)
    Kd = A.size(1)
    Md = hl.specialize(A.size(2))
    Nd = hl.specialize(B_w.size(0))

    output = torch.empty((Bat, Nd, Md), device=A.device, dtype=torch.bfloat16)

    for tb in hl.tile(Bat, block_size=1):
        for tm, tn in hl.tile([Md, Nd]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(Kd):
                a = A[tb.begin, tk, tm].to(torch.bfloat16)
                b = B_w[tn, tk].to(torch.bfloat16)
                acc = torch.addmm(acc, a.t(), b.t())

            acc = acc + bias[tn].to(torch.bfloat16)[None, :]
            output[tb.begin, tn, tm] = acc.to(torch.bfloat16).t()

    return output


def helion_fwd_view(A, B_w, bias, slice_size):
    out = helion_fwd_view_k(A, B_w, bias)
    return out, out[:, :slice_size, :]


# ═════════════════════════════════════════════════════════════════════════
# Harness
# ═════════════════════════════════════════════════════════════════════════
def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return t[len(t) // 2]


def check(out, sl, A, B_w, bias):
    ref_a = A.to(torch.bfloat16)
    ref_b = B_w.to(torch.bfloat16)
    ref_bias = bias.to(torch.bfloat16)
    ref = torch.matmul(ref_a.transpose(-2, -1), ref_b.T)
    ref = ref.transpose(-2, -1).contiguous()
    ref = (ref + ref_bias.unsqueeze(-1)).contiguous()
    out_ok = torch.allclose(out, ref, atol=0.5, rtol=0.1)
    sl_ok = torch.allclose(sl, ref[:, :SLICE, :], atol=0.5, rtol=0.1)
    sl_nz = sl.abs().sum().item() > 0
    return out_ok, sl_ok, sl_nz


def main():
    print(f"Forward benchmark: B={B}, M={M}, N={N}, K={K}, slice={SLICE}")
    print(f"transpose_A=True, transpose_C=True, 1D bias")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print(f"HELION_USE_DEFAULT_CONFIG={os.environ.get('HELION_USE_DEFAULT_CONFIG', 'unset')}\n")

    A = torch.randn(B, K, M, device="cuda", dtype=torch.float32)
    B_w = torch.randn(N, K, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    # ── Inductor ──
    print("Compiling Inductor forward...")
    o, s = inductor_fwd(A, B_w, bias, SLICE)
    torch.cuda.synchronize()
    ok = check(o, s, A, B_w, bias)
    t_ind = bench(inductor_fwd, A, B_w, bias, SLICE)
    print(f"  Inductor:       {t_ind:.4f} ms  correct={ok}")

    # ── Helion Original ──
    print("\nAutotuning Helion-Original forward...")
    o, s = helion_fwd_orig(A, B_w, bias, SLICE)
    torch.cuda.synchronize()
    ok = check(o, s, A, B_w, bias)
    t_orig = bench(helion_fwd_orig, A, B_w, bias, SLICE)
    print(f"  Helion-Orig:    {t_orig:.4f} ms  correct={ok}")

    # ── Helion View ──
    print("\nAutotuning Helion-View forward...")
    o, s = helion_fwd_view(A, B_w, bias, SLICE)
    torch.cuda.synchronize()
    ok = check(o, s, A, B_w, bias)
    t_view = bench(helion_fwd_view, A, B_w, bias, SLICE)
    print(f"  Helion-View:    {t_view:.4f} ms  correct={ok}")

    # ── Summary ──
    print("\n" + "=" * 55)
    print(f"  Inductor:       {t_ind:.4f} ms")
    print(f"  Helion-Orig:    {t_orig:.4f} ms  ({t_ind/t_orig:.2f}x vs Ind)")
    print(f"  Helion-View:    {t_view:.4f} ms  ({t_ind/t_view:.2f}x vs Ind)")
    print(f"  D96227354 ref:  Inductor=0.423ms, Helion=0.411ms")
    best = min(
        ("Inductor", t_ind), ("Helion-Orig", t_orig), ("Helion-View", t_view),
        key=lambda x: x[1],
    )
    print(f"  Winner: {best[0]}")


if __name__ == "__main__":
    main()
