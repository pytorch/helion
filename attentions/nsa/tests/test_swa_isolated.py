#!/usr/bin/env python
"""Isolated test for sliding window attention in NSA."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "swa" / "kernels"))

import torch
from flash_attention import FlashAttention
from parallel import flash_attn_func_local


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def naive_causal_sliding_window_attn(q, k, v, window_size, scale):
    """
    Naive PyTorch implementation of causal sliding window attention.
    q, k, v: (B, T, H, D)
    window_size: int - how many past tokens to attend to including current
    """
    B, T, H, D = q.shape
    D_v = v.shape[-1]

    # Scale queries
    q_scaled = q * scale

    # Compute attention for each position
    o = torch.zeros_like(v)
    for b in range(B):
        for t in range(T):
            # Window: [max(0, t - window_size + 1), t] inclusive
            start = max(0, t - window_size + 1)
            end = t + 1

            # Get key-value slice for the window
            k_window = k[b, start:end]  # (window_len, H, D)
            v_window = v[b, start:end]  # (window_len, H, D_v)

            # Compute attention scores: (H, 1, D) @ (H, D, window_len) -> (H, 1, window_len)
            q_t = q_scaled[b, t].unsqueeze(1)  # (H, 1, D)
            k_t = k_window.permute(1, 2, 0)  # (H, D, window_len)
            attn = torch.bmm(q_t, k_t).squeeze(1)  # (H, window_len)

            # Softmax
            attn = torch.softmax(attn, dim=-1)  # (H, window_len)

            # Compute output: (H, window_len) @ (H, window_len, D_v) -> (H, D_v)
            v_t = v_window.permute(1, 0, 2)  # (H, window_len, D_v)
            out = torch.bmm(attn.unsqueeze(1), v_t).squeeze(1)  # (H, D_v)

            o[b, t] = out

    return o


def test_sliding_window_forward():
    """Test forward pass only."""
    print("=" * 60)
    print("Test Sliding Window Forward")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H, D = 1, 64, 4, 32
    window_size = 16
    dtype = torch.float32
    scale = D ** -0.5

    q = torch.randn(B, T, H, D, dtype=dtype, device='cuda')
    k = torch.randn(B, T, H, D, dtype=dtype, device='cuda')
    v = torch.randn(B, T, H, D, dtype=dtype, device='cuda')

    # Reference: naive implementation
    ref = naive_causal_sliding_window_attn(q, k, v, window_size, scale)

    # Test: our flash_attn_func_local
    # window_size tuple: (left, right) where left=window_size-1 (how far back)
    tri = flash_attn_func_local(q, k, v, causal=True, window_size=(window_size-1, 0))

    ratio = get_err_ratio(ref, tri)
    print(f"Forward ratio: {ratio:.6f}")
    print(f"PASS: {ratio < 0.001}")
    return ratio < 0.001


def test_sliding_window_backward():
    """Test backward pass."""
    print("\n" + "=" * 60)
    print("Test Sliding Window Backward (No GQA)")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H, D = 1, 64, 4, 32
    window_size = 16
    dtype = torch.float32
    scale = D ** -0.5

    q = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H, D, dtype=dtype, device='cuda')

    # Reference: naive implementation with autograd
    ref = naive_causal_sliding_window_attn(q, k, v, window_size, scale)
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

    all_pass = all([
        get_err_ratio(ref, tri) < 0.001,
        get_err_ratio(ref_dq, tri_dq) < 0.01,
        get_err_ratio(ref_dk, tri_dk) < 0.01,
        get_err_ratio(ref_dv, tri_dv) < 0.01
    ])
    print(f"PASS: {all_pass}")
    return all_pass


def test_sliding_window_backward_gqa():
    """Test backward pass with GQA."""
    print("\n" + "=" * 60)
    print("Test Sliding Window Backward (With GQA)")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H_q, H_kv, D = 1, 64, 16, 4, 32
    window_size = 16
    dtype = torch.float32
    scale = D ** -0.5

    q = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda')

    # Reference: manually expand k, v and compute
    repeat_factor = H_q // H_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=2)
    v_expanded = v.repeat_interleave(repeat_factor, dim=2)

    ref = naive_causal_sliding_window_attn(q, k_expanded, v_expanded, window_size, scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Test: flash_attn_func_local (handles GQA internally)
    tri = flash_attn_func_local(q, k, v, causal=True, window_size=(window_size-1, 0))
    tri.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()

    print(f"Forward ratio: {get_err_ratio(ref, tri):.6f}")
    print(f"dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f"dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f"dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")

    all_pass = all([
        get_err_ratio(ref, tri) < 0.001,
        get_err_ratio(ref_dq, tri_dq) < 0.02,
        get_err_ratio(ref_dk, tri_dk) < 0.02,
        get_err_ratio(ref_dv, tri_dv) < 0.02
    ])
    print(f"PASS: {all_pass}")
    return all_pass


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    test_sliding_window_forward()
    test_sliding_window_backward()
    test_sliding_window_backward_gqa()
