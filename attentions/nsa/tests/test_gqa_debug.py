#!/usr/bin/env python
"""Debug test for GQA wrapper."""

import os
import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "swa" / "kernels"))

import torch
from flash_attention import FlashAttention
from parallel import FlashAttnGQAWrapper, flash_attn_func_local


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def test_no_gqa():
    """Test without GQA (H_q == H_kv)."""
    print("=" * 60)
    print("Test WITHOUT GQA (H_q == H_kv)")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H, D = 1, 64, 16, 32
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H, D, dtype=dtype, device='cuda')

    # Reference: direct FlashAttention with transposed tensors
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    ref_o_t = FlashAttention.apply(q_t, k_t, v_t, 32, "causal_sliding_window")
    ref_o = ref_o_t.transpose(1, 2).contiguous()

    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    # Test: FlashAttnGQAWrapper
    tri_o = FlashAttnGQAWrapper.apply(q, k, v, 32, "causal_sliding_window")
    tri_o.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()

    print(f"  o ratio: {get_err_ratio(ref_o, tri_o):.6f}")
    print(f" dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f" dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f" dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")

    all_pass = all([
        get_err_ratio(ref_o, tri_o) < 0.001,
        get_err_ratio(ref_dq, tri_dq) < 0.001,
        get_err_ratio(ref_dk, tri_dk) < 0.001,
        get_err_ratio(ref_dv, tri_dv) < 0.001
    ])
    print(f"PASS: {all_pass}")
    return all_pass


def test_with_gqa():
    """Test with GQA (H_q > H_kv)."""
    print("\n" + "=" * 60)
    print("Test WITH GQA (H_q = 16, H_kv = 4)")
    print("=" * 60)
    torch.manual_seed(42)

    B, T, H_q, H_kv, D = 1, 64, 16, 4, 32
    dtype = torch.bfloat16

    q = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H_kv, D, dtype=dtype, device='cuda', requires_grad=True)
    do = torch.randn(B, T, H_q, D, dtype=dtype, device='cuda')

    # Reference: manually expand k, v and use FlashAttention directly
    repeat_factor = H_q // H_kv
    k_expanded = k.repeat_interleave(repeat_factor, dim=2)
    v_expanded = v.repeat_interleave(repeat_factor, dim=2)

    q_t = q.transpose(1, 2).contiguous()
    k_t = k_expanded.transpose(1, 2).contiguous()
    v_t = v_expanded.transpose(1, 2).contiguous()

    ref_o_t = FlashAttention.apply(q_t, k_t, v_t, 32, "causal_sliding_window")
    ref_o = ref_o_t.transpose(1, 2).contiguous()

    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    print(f"Reference shapes: dq={ref_dq.shape}, dk={ref_dk.shape}, dv={ref_dv.shape}")

    # Test: FlashAttnGQAWrapper
    tri_o = FlashAttnGQAWrapper.apply(q, k, v, 32, "causal_sliding_window")
    tri_o.backward(do)
    tri_dq = q.grad.clone()
    tri_dk = k.grad.clone()
    tri_dv = v.grad.clone()

    print(f"Triton shapes: dq={tri_dq.shape}, dk={tri_dk.shape}, dv={tri_dv.shape}")

    print(f"  o ratio: {get_err_ratio(ref_o, tri_o):.6f}")
    print(f" dq ratio: {get_err_ratio(ref_dq, tri_dq):.6f}")
    print(f" dk ratio: {get_err_ratio(ref_dk, tri_dk):.6f}")
    print(f" dv ratio: {get_err_ratio(ref_dv, tri_dv):.6f}")

    all_pass = all([
        get_err_ratio(ref_o, tri_o) < 0.001,
        get_err_ratio(ref_dq, tri_dq) < 0.05,  # More tolerance for gradients
        get_err_ratio(ref_dk, tri_dk) < 0.05,
        get_err_ratio(ref_dv, tri_dv) < 0.05
    ])
    print(f"PASS: {all_pass}")
    return all_pass


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    test_no_gqa()
    test_with_gqa()
