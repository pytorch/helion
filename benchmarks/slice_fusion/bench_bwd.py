"""
Backward kernel benchmark for matmul + slice fusion (LCE pattern).

Shape: (B=4096, M=128, N=192, K=48, slice_size=96)
transpose_A=True, transpose_C=True, 1D bias.

Forward: output[B,N,M] = (A[B,K,M]^T @ B_w[N,K]^T + bias[N,None]).contiguous()
Slice:   sliced = output[:, :slice_size, :]

Backward math:
  d_eff = d_out;  d_eff[:, :slice, :] += d_sliced
  d_A  = (d_eff.transpose(-2,-1) @ B_w).transpose(-2,-1)   # [B,K,M]
  d_Bw = (A @ d_eff.transpose(-2,-1)).sum(0).T              # [N,K]
  d_bias = d_eff.sum(dim=(0,2))                              # [N]

Compares:
  1. Inductor (torch.compile)
  2. Helion-Original: inline d_sliced accumulation in kernel
  3. Helion-View: pre-add d_sliced to d_out, then standard kernel

Context: https://github.com/pytorch/helion/issues/1679

Usage:
  # On H100 with denoising:
  CUDA_VISIBLE_DEVICES=7 triton/scripts/denoise.sh python benchmarks/slice_fusion/bench_bwd.py

  # Quick run (skip autotuning, use cached/default configs):
  CUDA_VISIBLE_DEVICES=7 HELION_USE_DEFAULT_CONFIG=1 python benchmarks/slice_fusion/bench_bwd.py
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
def inductor_bwd(d_out, d_sliced, A, B_w, slice_size):
    """Full backward with inline slice accumulation via torch.compile."""
    d_eff = d_out.clone()
    d_eff[:, :slice_size, :] += d_sliced
    d_C = d_eff.transpose(-2, -1)                          # [B,M,N]
    d_A = torch.bmm(d_C, B_w.unsqueeze(0).expand(d_C.size(0), -1, -1))  # [B,M,K]
    d_A = d_A.transpose(-2, -1)                            # [B,K,M]
    d_Bw = torch.bmm(A, d_C).sum(0).T                     # [N,K]
    d_bias = d_eff.sum(dim=(0, 2))                         # [N]
    return d_A, d_Bw, d_bias


# ═════════════════════════════════════════════════════════════════════════
# 2. Helion-Original: inline d_sliced in kernel
# ═════════════════════════════════════════════════════════════════════════
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def helion_bwd_orig(
    d_out: torch.Tensor,       # [B, N, M]
    d_sliced: torch.Tensor,    # [B, slice, M]
    A: torch.Tensor,           # [B, K, M]
    B_w: torch.Tensor,         # [N, K]
    slice_size: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Bat = d_out.size(0)
    Nd = hl.specialize(d_out.size(1))
    Md = hl.specialize(d_out.size(2))
    Kd = A.size(1)

    d_A = torch.empty((Bat, Kd, Md), device=A.device, dtype=A.dtype)
    d_Bw = torch.zeros((Nd, Kd), device=B_w.device, dtype=torch.float32)
    d_bias = torch.zeros((Bat, Nd), device=A.device, dtype=torch.float32)

    for tb in hl.tile(Bat, block_size=1):
        # ── d_A: [M,N]@[N,K] -> [M,K] -> transpose -> [K,M] ──
        for tm, tk in hl.tile([Md, Kd]):
            acc_dA = hl.zeros([tm, tk], dtype=torch.float32)
            for tn in hl.tile(Nd):
                dout_tile = d_out[tb.begin, tn, tm]       # [tn, tm]
                if tn.end <= slice_size:
                    ds_tile = d_sliced[tb.begin, tn, tm]  # [tn, tm]
                    dout_tile = dout_tile + ds_tile
                bw_tile = B_w[tn, tk]                     # [tn, tk]
                acc_dA = torch.addmm(acc_dA, dout_tile.t(), bw_tile)
            d_A[tb.begin, tk, tm] = acc_dA.t()

        # ── d_Bw: [K,M]@[M,N] -> [K,N] -> atomic add ──
        for tn, tk in hl.tile([Nd, Kd]):
            acc_dB = hl.zeros([tk, tn], dtype=torch.float32)
            for tm in hl.tile(Md):
                dout_tile = d_out[tb.begin, tn, tm]       # [tn, tm]
                if tn.end <= slice_size:
                    ds_tile = d_sliced[tb.begin, tn, tm]
                    dout_tile = dout_tile + ds_tile
                a_tile = A[tb.begin, tk, tm]               # [tk, tm]
                acc_dB = torch.addmm(acc_dB, a_tile, dout_tile.t())
            hl.atomic_add(d_Bw, [tn, tk], acc_dB.t())

        # ── d_bias: sum d_eff over M ──
        for tn in hl.tile(Nd):
            acc_db = hl.zeros([tn], dtype=torch.float32)
            for tm in hl.tile(Md):
                dout_tile = d_out[tb.begin, tn, tm]       # [tn, tm]
                if tn.end <= slice_size:
                    ds_tile = d_sliced[tb.begin, tn, tm]
                    dout_tile = dout_tile + ds_tile
                acc_db = acc_db + dout_tile.sum(dim=1)
            d_bias[tb.begin, tn] = acc_db

    return d_A, d_Bw, d_bias.sum(dim=0)


# ═════════════════════════════════════════════════════════════════════════
# 3. Helion-View: pre-add d_sliced, then standard kernel (no slice logic)
# ═════════════════════════════════════════════════════════════════════════
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def helion_bwd_view_k(
    d_eff: torch.Tensor,       # [B, N, M] (pre-accumulated)
    A: torch.Tensor,           # [B, K, M]
    B_w: torch.Tensor,         # [N, K]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Bat = d_eff.size(0)
    Nd = hl.specialize(d_eff.size(1))
    Md = hl.specialize(d_eff.size(2))
    Kd = A.size(1)

    d_A = torch.empty((Bat, Kd, Md), device=A.device, dtype=A.dtype)
    d_Bw = torch.zeros((Nd, Kd), device=B_w.device, dtype=torch.float32)
    d_bias = torch.zeros((Bat, Nd), device=A.device, dtype=torch.float32)

    for tb in hl.tile(Bat, block_size=1):
        # ── d_A ──
        for tm, tk in hl.tile([Md, Kd]):
            acc_dA = hl.zeros([tm, tk], dtype=torch.float32)
            for tn in hl.tile(Nd):
                deff_tile = d_eff[tb.begin, tn, tm]       # [tn, tm]
                bw_tile = B_w[tn, tk]                     # [tn, tk]
                acc_dA = torch.addmm(acc_dA, deff_tile.t(), bw_tile)
            d_A[tb.begin, tk, tm] = acc_dA.t()

        # ── d_Bw ──
        for tn, tk in hl.tile([Nd, Kd]):
            acc_dB = hl.zeros([tk, tn], dtype=torch.float32)
            for tm in hl.tile(Md):
                deff_tile = d_eff[tb.begin, tn, tm]       # [tn, tm]
                a_tile = A[tb.begin, tk, tm]               # [tk, tm]
                acc_dB = torch.addmm(acc_dB, a_tile, deff_tile.t())
            hl.atomic_add(d_Bw, [tn, tk], acc_dB.t())

        # ── d_bias ──
        for tn in hl.tile(Nd):
            acc_db = hl.zeros([tn], dtype=torch.float32)
            for tm in hl.tile(Md):
                deff_tile = d_eff[tb.begin, tn, tm]       # [tn, tm]
                acc_db = acc_db + deff_tile.sum(dim=1)
            d_bias[tb.begin, tn] = acc_db

    return d_A, d_Bw, d_bias.sum(dim=0)


def helion_bwd_view(d_out, d_sliced, A, B_w, slice_size):
    d_eff = d_out.clone()
    d_eff[:, :slice_size, :] += d_sliced
    return helion_bwd_view_k(d_eff, A, B_w)


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


def check(d_A, d_Bw, d_bias, d_out, d_sliced, A, B_w):
    d_eff = d_out.clone().float()
    d_eff[:, :SLICE, :] += d_sliced.float()
    d_C = d_eff.transpose(-2, -1)
    ref_dA = torch.bmm(d_C, B_w.float().unsqueeze(0).expand(B, -1, -1)).transpose(-2, -1)
    ref_dBw = torch.bmm(A.float(), d_C).sum(0).T
    ref_dbias = d_eff.sum(dim=(0, 2))
    ok_dA = torch.allclose(d_A.float(), ref_dA, atol=1.0, rtol=0.1)
    ok_dBw = torch.allclose(d_Bw.float(), ref_dBw, atol=5.0, rtol=0.1)
    ok_db = torch.allclose(d_bias.float(), ref_dbias, atol=5.0, rtol=0.1)
    return ok_dA, ok_dBw, ok_db


def main():
    print(f"Backward benchmark: B={B}, M={M}, N={N}, K={K}, slice={SLICE}")
    print(f"transpose_A=True, transpose_C=True, 1D bias")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print(f"HELION_USE_DEFAULT_CONFIG={os.environ.get('HELION_USE_DEFAULT_CONFIG', 'unset')}\n")

    d_out = torch.randn(B, N, M, device="cuda", dtype=torch.bfloat16)
    d_sliced = torch.randn(B, SLICE, M, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(B, K, M, device="cuda", dtype=torch.bfloat16)
    B_w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # ── Inductor ──
    print("Compiling Inductor backward...")
    dA, dBw, db = inductor_bwd(d_out, d_sliced, A, B_w, SLICE)
    torch.cuda.synchronize()
    ok = check(dA, dBw, db, d_out, d_sliced, A, B_w)
    t_ind = bench(inductor_bwd, d_out, d_sliced, A, B_w, SLICE)
    print(f"  Inductor:       {t_ind:.4f} ms  correct={ok}")

    # ── Helion Original ──
    print("\nAutotuning Helion-Original backward...")
    dA, dBw, db = helion_bwd_orig(d_out, d_sliced, A, B_w, SLICE)
    torch.cuda.synchronize()
    ok = check(dA, dBw, db, d_out, d_sliced, A, B_w)
    t_orig = bench(helion_bwd_orig, d_out, d_sliced, A, B_w, SLICE)
    print(f"  Helion-Orig:    {t_orig:.4f} ms  correct={ok}")

    # ── Helion View ──
    print("\nAutotuning Helion-View backward...")
    dA, dBw, db = helion_bwd_view(d_out, d_sliced, A, B_w, SLICE)
    torch.cuda.synchronize()
    ok = check(dA, dBw, db, d_out, d_sliced, A, B_w)
    t_view = bench(helion_bwd_view, d_out, d_sliced, A, B_w, SLICE)
    print(f"  Helion-View:    {t_view:.4f} ms  correct={ok}")

    # ── Summary ──
    print("\n" + "=" * 55)
    print(f"  Inductor:       {t_ind:.4f} ms")
    print(f"  Helion-Orig:    {t_orig:.4f} ms  ({t_ind/t_orig:.2f}x vs Ind)")
    print(f"  Helion-View:    {t_view:.4f} ms  ({t_ind/t_view:.2f}x vs Ind)")
    print(f"  D96227354 ref:  Inductor=0.699ms, Helion=1.108ms")
    best = min(
        ("Inductor", t_ind), ("Helion-Orig", t_orig), ("Helion-View", t_view),
        key=lambda x: x[1],
    )
    print(f"  Winner: {best[0]}")


if __name__ == "__main__":
    main()
