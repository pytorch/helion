#!/usr/bin/env python
"""Test selected attention only (no sliding window)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch
import triton

from naive import naive_nsa
from parallel import parallel_nsa


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def test_selected_only():
    """Test NSA without sliding window."""
    print("=" * 60)
    print("Test NSA - Selected Attention Only (no sliding window)")
    print("=" * 60)
    torch.manual_seed(42)

    B = 1
    T = 128
    H = 4
    HQ = 64
    D = 64
    S = 16
    block_size = 32
    window_size = 0  # No sliding window!
    dtype = torch.bfloat16
    scale = 0.1

    print(f"B={B}, T={T}, H={H}, HQ={HQ}, D={D}, S={S}")

    perm_q = torch.randperm(T, device='cuda')
    perm_k = torch.randperm(T, device='cuda')
    perm_v = torch.randperm(T, device='cuda')
    q = torch.linspace(0, 1, steps=T, dtype=dtype, device='cuda')[perm_q].view(1, T, 1, 1).expand(B, T, HQ, D).clone().requires_grad_(True)
    k = torch.linspace(0, 1, steps=T, dtype=dtype, device='cuda')[perm_k].view(1, T, 1, 1).expand(B, T, H, D).clone().requires_grad_(True)
    v = torch.linspace(0, 1, steps=T, dtype=dtype, device='cuda')[perm_v].view(1, T, 1, 1).expand(B, T, H, D).clone().requires_grad_(True)
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, T, H), dtype=torch.long, device='cuda')

    print("Running naive reference...")
    ref = naive_nsa(
        q=q, k=k, v=v, g_slc=g_slc, g_swa=g_swa,
        block_indices=block_indices, block_counts=block_counts,
        block_size=block_size, window_size=window_size, scale=scale
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    print("Running parallel Triton implementation...")
    tri = parallel_nsa(
        q=q, k=k, v=v, g_slc=g_slc, g_swa=g_swa,
        block_indices=block_indices, block_counts=block_counts,
        block_size=block_size, window_size=window_size, scale=scale
    )
    tri.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()
    tri_dg_slc = g_slc.grad.clone()

    print(f"  o ratio: {get_err_ratio(ref, tri):.6f}")
    print(f" dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f" dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f" dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")
    print(f"dg_slc ratio: {get_err_ratio(ref_dg_slc, tri_dg_slc):.6f}")

    all_pass = all([
        get_err_ratio(ref, tri) < 0.01,
        get_err_ratio(ref_dq, tri_dq) < 0.01,
        get_err_ratio(ref_dk, tri_dk) < 0.01,
        get_err_ratio(ref_dv, tri_dv) < 0.01,
        get_err_ratio(ref_dg_slc, tri_dg_slc) < 0.01,
    ])
    print(f"PASS: {all_pass}")
    return all_pass


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    test_selected_only()
