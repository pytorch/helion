#!/usr/bin/env python
"""Test to understand precision differences."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "swa" / "kernels"))

import torch
import triton

from parallel import flash_attn_func_local


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def naive_swa_float32(q, k, v, window_size, scale):
    """Naive sliding window in float32."""
    B, T, H, D = q.shape

    # Convert to float32 like naive_nsa does
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    o = torch.zeros(B, T, H, D, dtype=torch.float32, device=q.device)

    for b in range(B):
        for t in range(T):
            start = max(0, t - window_size + 1)
            end = t + 1

            q_t = q_f[b, t] * scale  # (H, D)
            k_window = k_f[b, start:end]  # (window_len, H, D)
            v_window = v_f[b, start:end]  # (window_len, H, D_v)

            # (H, D) @ (H, D, window_len) -> (H, window_len)
            attn = torch.einsum('h d, n h d -> h n', q_t, k_window)
            attn = torch.softmax(attn, dim=-1)

            # (H, window_len) @ (H, window_len, D) -> (H, D)
            out = torch.einsum('h n, n h d -> h d', attn, v_window)
            o[b, t] = out

    return o.to(q.dtype)


def test_with_gqa():
    """Test precision with GQA."""
    print("=" * 60)
    print("Test Precision with GQA")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H_q, H_kv, D = 1, 64, 64, 4, 64
    window_size = 32
    dtype = torch.bfloat16
    scale = D ** -0.5

    q = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda')

    # Reference: expand k, v and use float32 naive
    repeat_factor = H_q // H_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=2)
    v_expanded = v.repeat_interleave(repeat_factor, dim=2)

    ref = naive_swa_float32(q, k_expanded, v_expanded, window_size, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Test: flash_attn_func_local
    tri = flash_attn_func_local(q, k, v, causal=True, window_size=(window_size-1, 0))
    tri.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()

    print(f"Forward ratio: {get_err_ratio(ref, tri):.6f}")
    print(f"dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f"dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f"dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")


def test_same_precision():
    """Test with same precision."""
    print("\n" + "=" * 60)
    print("Test Same Precision (float32)")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H_q, H_kv, D = 1, 64, 64, 4, 64
    window_size = 32
    dtype = torch.float32  # Use float32
    scale = D ** -0.5

    q = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda')

    # Reference: expand k, v and use float32 naive
    repeat_factor = H_q // H_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=2)
    v_expanded = v.repeat_interleave(repeat_factor, dim=2)

    ref = naive_swa_float32(q, k_expanded, v_expanded, window_size, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Test: flash_attn_func_local
    tri = flash_attn_func_local(q, k, v, causal=True, window_size=(window_size-1, 0))
    tri.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()

    print(f"Forward ratio: {get_err_ratio(ref, tri):.6f}")
    print(f"dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f"dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f"dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    test_with_gqa()
    test_same_precision()
