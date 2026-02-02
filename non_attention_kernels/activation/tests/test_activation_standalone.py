# SPDX-License-Identifier: Apache-2.0
"""
Standalone tests for activation kernels with PyTorch reference implementations.
"""
import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch
import torch.nn.functional as F

from k_activations_mamba import _swiglu_fwd, swiglu


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def swiglu_ref(xy):
    """Reference implementation of SwiGLU using PyTorch."""
    x, y = xy.chunk(2, dim=-1)
    return F.silu(x) * y


def swiglu_bwd_ref(xy, dout):
    """Reference implementation of SwiGLU backward using PyTorch."""
    xy = xy.clone().requires_grad_(True)
    x, y = xy.chunk(2, dim=-1)
    out = F.silu(x) * y
    out.backward(dout)
    return xy.grad


class TestSwiGLU:
    """Tests for SwiGLU activation kernel."""

    def test_swiglu_forward_basic(self):
        """Test basic SwiGLU forward pass."""
        batch_size = 4
        hidden_dim = 256

        xy = torch.randn(batch_size, hidden_dim * 2, device="cuda", dtype=torch.float32)

        # Triton kernel
        out_triton = _swiglu_fwd(xy)

        # Reference
        out_ref = swiglu_ref(xy)

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-3, atol=1e-3)

    def test_swiglu_forward_various_shapes(self):
        """Test SwiGLU forward with various input shapes."""
        shapes = [
            (1, 64),
            (8, 128),
            (32, 512),
            (16, 1024),
            (4, 8, 256),  # 3D input
        ]

        for shape in shapes:
            xy = torch.randn(*shape[:-1], shape[-1] * 2, device="cuda", dtype=torch.float32)
            out_triton = _swiglu_fwd(xy)
            out_ref = swiglu_ref(xy)
            torch.testing.assert_close(
                out_triton, out_ref, rtol=1e-3, atol=1e-3,
                msg=f"Failed for shape {shape}"
            )

    def test_swiglu_autograd(self):
        """Test SwiGLU with autograd."""
        batch_size = 4
        hidden_dim = 128

        xy = torch.randn(batch_size, hidden_dim * 2, device="cuda", dtype=torch.float32, requires_grad=True)
        xy_ref = xy.clone().detach().requires_grad_(True)

        # Forward
        out = swiglu(xy)
        out_ref = swiglu_ref(xy_ref)

        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)

        # Backward
        dout = torch.randn_like(out)
        out.backward(dout)
        out_ref.backward(dout)

        torch.testing.assert_close(xy.grad, xy_ref.grad, rtol=1e-2, atol=1e-2)

    def test_swiglu_dtypes(self):
        """Test SwiGLU with different dtypes."""
        batch_size = 8
        hidden_dim = 256

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            xy = torch.randn(batch_size, hidden_dim * 2, device="cuda", dtype=dtype)
            out_triton = _swiglu_fwd(xy)
            out_ref = swiglu_ref(xy.float()).to(dtype)

            # Looser tolerance for lower precision
            rtol = 1e-2 if dtype != torch.float32 else 1e-3
            atol = 1e-2 if dtype != torch.float32 else 1e-3

            torch.testing.assert_close(
                out_triton, out_ref, rtol=rtol, atol=atol,
                msg=f"Failed for dtype {dtype}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
