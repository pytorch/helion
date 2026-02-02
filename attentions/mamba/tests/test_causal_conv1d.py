#!/usr/bin/env python
"""Test local causal_conv1d implementation."""

import sys
from pathlib import Path

# Add local kernels path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch
import torch.nn.functional as F

from causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_fwd_function,
    causal_conv1d_bwd_function,
)


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """Reference implementation using PyTorch."""
    batch, dim, seqlen = x.shape
    _, width = weight.shape

    # Pad on the left for causal convolution
    x_padded = F.pad(x, (width - 1, 0))

    # Apply depthwise convolution
    weight_reshaped = weight.unsqueeze(1)  # (dim, 1, width)
    out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)

    if activation in ["silu", "swish"]:
        out = F.silu(out)

    return out


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = x.flatten().square().mean().sqrt().item()
    return err / base if base > 0 else err


def test_forward():
    """Test forward pass."""
    print("=" * 60)
    print("Test causal_conv1d forward pass")
    print("=" * 60)

    torch.manual_seed(42)

    batch, dim, seqlen = 2, 64, 128
    width = 4
    dtype = torch.float32
    device = 'cuda'

    x = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)

    # Reference
    ref = causal_conv1d_ref(x, weight, bias, activation="silu")

    # Our implementation
    out = causal_conv1d_fn(x, weight, bias, activation="silu")

    ratio = get_err_ratio(ref, out)
    print(f"Forward ratio: {ratio:.6f}")
    print(f"PASS: {ratio < 0.001}")
    return ratio < 0.001


def test_forward_no_activation():
    """Test forward pass without activation."""
    print("\n" + "=" * 60)
    print("Test causal_conv1d forward (no activation)")
    print("=" * 60)

    torch.manual_seed(42)

    batch, dim, seqlen = 2, 64, 128
    width = 4
    dtype = torch.float32
    device = 'cuda'

    x = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)
    weight = torch.randn(dim, width, dtype=dtype, device=device)
    bias = torch.randn(dim, dtype=dtype, device=device)

    # Reference
    ref = causal_conv1d_ref(x, weight, bias, activation=None)

    # Our implementation
    out = causal_conv1d_fn(x, weight, bias, activation=None)

    ratio = get_err_ratio(ref, out)
    print(f"Forward ratio: {ratio:.6f}")
    print(f"PASS: {ratio < 0.001}")
    return ratio < 0.001


def test_backward():
    """Test backward pass."""
    print("\n" + "=" * 60)
    print("Test causal_conv1d backward pass")
    print("=" * 60)

    torch.manual_seed(42)

    batch, dim, seqlen = 2, 64, 128
    width = 4
    dtype = torch.float32
    device = 'cuda'

    x = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(dim, width, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn(batch, dim, seqlen, dtype=dtype, device=device)

    # Reference backward
    ref = causal_conv1d_ref(x, weight, bias, activation="silu")
    ref.backward(dout)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dweight, weight.grad = weight.grad.clone(), None
    ref_dbias, bias.grad = bias.grad.clone(), None

    # Our implementation backward
    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    out.backward(dout)
    tri_dx = x.grad.clone()
    tri_dweight = weight.grad.clone()
    tri_dbias = bias.grad.clone()

    print(f"Forward ratio: {get_err_ratio(ref, out):.6f}")
    print(f"dx ratio: {get_err_ratio(ref_dx, tri_dx):.6f}")
    print(f"dweight ratio: {get_err_ratio(ref_dweight, tri_dweight):.6f}")
    print(f"dbias ratio: {get_err_ratio(ref_dbias, tri_dbias):.6f}")

    all_pass = all([
        get_err_ratio(ref, out) < 0.001,
        get_err_ratio(ref_dx, tri_dx) < 0.01,
        get_err_ratio(ref_dweight, tri_dweight) < 0.01,
        get_err_ratio(ref_dbias, tri_dbias) < 0.01,
    ])
    print(f"PASS: {all_pass}")
    return all_pass


def test_import_from_ssd_combined():
    """Test that ssd_combined can import from local causal_conv1d."""
    print("\n" + "=" * 60)
    print("Test import from ssd_combined")
    print("=" * 60)

    try:
        from ssd_combined import mamba_chunk_scan_combined
        print("Import successful!")
        print("PASS: True")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        print("PASS: False")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    results = []
    results.append(test_forward())
    results.append(test_forward_no_activation())
    results.append(test_backward())
    results.append(test_import_from_ssd_combined())

    print("\n" + "=" * 60)
    print(f"All tests passed: {all(results)}")
    print("=" * 60)
