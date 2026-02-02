# SPDX-License-Identifier: Apache-2.0
"""
Standalone tests for normalization kernels with PyTorch reference implementations.
"""
import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch
import torch.nn.functional as F

from layernorm_gated_mamba import rmsnorm_fn, layernorm_fn


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def rmsnorm_ref(x, weight, eps=1e-6):
    """Reference implementation of RMSNorm using PyTorch."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * weight


def layernorm_ref(x, weight, bias, eps=1e-6):
    """Reference implementation of LayerNorm using PyTorch."""
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps)


def gated_rmsnorm_ref(x, z, weight, eps=1e-6, norm_before_gate=True):
    """Reference implementation of gated RMSNorm."""
    if norm_before_gate:
        norm_out = rmsnorm_ref(x, weight, eps)
        return norm_out * F.silu(z)
    else:
        gated = x * F.silu(z)
        return rmsnorm_ref(gated, weight, eps)


class TestRMSNorm:
    """Tests for RMSNorm kernel."""

    def test_rmsnorm_forward_basic(self):
        """Test basic RMSNorm forward pass."""
        batch_size = 4
        hidden_dim = 256

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)

        # Triton kernel
        out_triton = rmsnorm_fn(x, weight, None)

        # Reference
        out_ref = rmsnorm_ref(x, weight)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)

    def test_rmsnorm_forward_various_shapes(self):
        """Test RMSNorm forward with various input shapes."""
        shapes = [
            (1, 64),
            (8, 128),
            (32, 512),
            (16, 1024),
        ]

        for shape in shapes:
            x = torch.randn(*shape, device="cuda", dtype=torch.float32)
            weight = torch.ones(shape[-1], device="cuda", dtype=torch.float32)

            out_triton = rmsnorm_fn(x, weight, None)
            out_ref = rmsnorm_ref(x, weight)

            torch.testing.assert_close(
                out_triton, out_ref, rtol=1e-3, atol=1e-3,
                msg=f"Failed for shape {shape}"
            )


class TestLayerNorm:
    """Tests for LayerNorm kernel."""

    def test_layernorm_forward_basic(self):
        """Test basic LayerNorm forward pass."""
        batch_size = 4
        hidden_dim = 256

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden_dim, device="cuda", dtype=torch.float32)

        # Triton kernel
        out_triton = layernorm_fn(x, weight, bias)

        # Reference
        out_ref = layernorm_ref(x, weight, bias)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)

    def test_layernorm_with_bias(self):
        """Test LayerNorm with non-zero bias."""
        batch_size = 8
        hidden_dim = 512

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

        out_triton = layernorm_fn(x, weight, bias)
        out_ref = layernorm_ref(x, weight, bias)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)


class TestGatedNorm:
    """Tests for gated normalization kernels."""

    def test_gated_rmsnorm_forward(self):
        """Test gated RMSNorm forward pass."""
        batch_size = 4
        hidden_dim = 256

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        z = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)

        # Triton kernel with gating
        out_triton = rmsnorm_fn(x, weight, None, z=z, norm_before_gate=True)

        # Reference
        out_ref = gated_rmsnorm_ref(x, z, weight, norm_before_gate=True)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)

    def test_gated_rmsnorm_norm_after_gate(self):
        """Test gated RMSNorm with norm applied after gating."""
        batch_size = 4
        hidden_dim = 256

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        z = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float32)

        # Triton kernel with gating (norm after gate)
        out_triton = rmsnorm_fn(x, weight, None, z=z, norm_before_gate=False)

        # Reference
        out_ref = gated_rmsnorm_ref(x, z, weight, norm_before_gate=False)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)


class TestNormAutograd:
    """Tests for normalization autograd support."""

    def test_rmsnorm_backward(self):
        """Test RMSNorm backward pass."""
        batch_size = 4
        hidden_dim = 128

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_(True)
        weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float32, requires_grad=True)
        weight_ref = weight.clone().detach().requires_grad_(True)

        # Forward
        out = rmsnorm_fn(x, weight, None)
        out_ref = rmsnorm_ref(x_ref, weight_ref)

        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)

        # Backward
        dout = torch.randn_like(out)
        out.backward(dout)
        out_ref.backward(dout)

        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
