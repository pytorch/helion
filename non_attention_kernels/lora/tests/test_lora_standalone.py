# SPDX-License-Identifier: Apache-2.0
"""
Standalone tests for LoRA (Low-Rank Adaptation) kernels with PyTorch reference implementations.
"""
import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

try:
    from lora_shrink_vllm import _lora_shrink_kernel, lora_shrink
    HAS_LORA_SHRINK = True
except ImportError:
    HAS_LORA_SHRINK = False
try:
    from lora_expand_vllm import _lora_expand_kernel, lora_expand
    HAS_LORA_EXPAND = True
except ImportError:
    HAS_LORA_EXPAND = False


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def lora_shrink_ref(x, lora_a, scale=1.0):
    """
    Reference implementation of LoRA shrink (A matrix multiplication).

    LoRA shrink projects from hidden_dim to lora_rank:
    output = x @ A.T * scale

    Args:
        x: Input tensor [batch, hidden_dim]
        lora_a: LoRA A matrix [lora_rank, hidden_dim]
        scale: Scaling factor

    Returns:
        Tensor [batch, lora_rank]
    """
    return x @ lora_a.T * scale


def lora_expand_ref(x, lora_b, scale=1.0):
    """
    Reference implementation of LoRA expand (B matrix multiplication).

    LoRA expand projects from lora_rank to hidden_dim:
    output = x @ B.T * scale

    Args:
        x: Input tensor [batch, lora_rank]
        lora_b: LoRA B matrix [hidden_dim, lora_rank]
        scale: Scaling factor

    Returns:
        Tensor [batch, hidden_dim]
    """
    return x @ lora_b.T * scale


def lora_forward_ref(x, lora_a, lora_b, scale=1.0):
    """
    Reference implementation of full LoRA forward pass.

    LoRA: output = x @ A.T @ B.T * scale = x @ (BA).T * scale

    Args:
        x: Input tensor [batch, hidden_dim]
        lora_a: LoRA A matrix [lora_rank, hidden_dim]
        lora_b: LoRA B matrix [output_dim, lora_rank]
        scale: Scaling factor (typically alpha / rank)

    Returns:
        Tensor [batch, output_dim]
    """
    intermediate = x @ lora_a.T
    return intermediate @ lora_b.T * scale


def batched_lora_forward_ref(x, lora_a_stacked, lora_b_stacked, indices, scale=1.0):
    """
    Reference implementation of batched LoRA forward.

    Different samples can use different LoRA adapters.

    Args:
        x: Input tensor [batch, hidden_dim]
        lora_a_stacked: Stacked A matrices [num_loras, lora_rank, hidden_dim]
        lora_b_stacked: Stacked B matrices [num_loras, output_dim, lora_rank]
        indices: LoRA index per sample [batch]
        scale: Scaling factor

    Returns:
        Tensor [batch, output_dim]
    """
    batch_size = x.shape[0]
    output_dim = lora_b_stacked.shape[1]
    output = torch.zeros(batch_size, output_dim, device=x.device, dtype=x.dtype)

    for i in range(batch_size):
        lora_idx = indices[i].item()
        if lora_idx >= 0:  # -1 means no LoRA
            lora_a = lora_a_stacked[lora_idx]
            lora_b = lora_b_stacked[lora_idx]
            output[i] = lora_forward_ref(x[i:i+1], lora_a, lora_b, scale).squeeze(0)

    return output


class TestLoRAShrink:
    """Tests for LoRA shrink (A matrix) kernel."""

    @pytest.mark.skipif(not HAS_LORA_SHRINK, reason="lora_shrink_vllm not available")
    def test_lora_shrink_basic(self):
        """Test basic LoRA shrink operation."""
        batch_size = 8
        hidden_dim = 256
        lora_rank = 16

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
        lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float16)

        out_triton = lora_shrink(x, lora_a)
        out_ref = lora_shrink_ref(x.float(), lora_a.float()).half()

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_LORA_SHRINK, reason="lora_shrink_vllm not available")
    def test_lora_shrink_various_ranks(self):
        """Test LoRA shrink with various ranks."""
        batch_size = 8
        hidden_dim = 256

        for lora_rank in [8, 16, 32, 64]:
            x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
            lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float16)

            out_triton = lora_shrink(x, lora_a)
            assert out_triton.shape == (batch_size, lora_rank), f"Wrong shape for rank {lora_rank}"


class TestLoRAExpand:
    """Tests for LoRA expand (B matrix) kernel."""

    @pytest.mark.skipif(not HAS_LORA_EXPAND, reason="lora_expand_vllm not available")
    def test_lora_expand_basic(self):
        """Test basic LoRA expand operation."""
        batch_size = 8
        lora_rank = 16
        output_dim = 256

        x = torch.randn(batch_size, lora_rank, device="cuda", dtype=torch.float16)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float16)

        out_triton = lora_expand(x, lora_b)
        out_ref = lora_expand_ref(x.float(), lora_b.float()).half()

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not HAS_LORA_EXPAND, reason="lora_expand_vllm not available")
    def test_lora_expand_with_add(self):
        """Test LoRA expand with add to existing tensor."""
        batch_size = 8
        lora_rank = 16
        output_dim = 256

        x = torch.randn(batch_size, lora_rank, device="cuda", dtype=torch.float16)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float16)
        base_output = torch.randn(batch_size, output_dim, device="cuda", dtype=torch.float16)

        # Some implementations add to existing output
        lora_out = lora_expand(x, lora_b)
        combined = base_output + lora_out

        ref_lora = lora_expand_ref(x.float(), lora_b.float()).half()
        ref_combined = base_output + ref_lora

        torch.testing.assert_close(combined, ref_combined, rtol=1e-2, atol=1e-2)


class TestLoRAFull:
    """Tests for full LoRA forward pass."""

    @pytest.mark.skipif(not HAS_LORA_SHRINK or not HAS_LORA_EXPAND, reason="LoRA kernels not available")
    def test_lora_full_forward(self):
        """Test full LoRA forward: x @ A.T @ B.T * scale."""
        batch_size = 8
        hidden_dim = 256
        lora_rank = 16
        output_dim = 512
        scale = 0.5

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
        lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float16)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float16)

        # Two-step LoRA
        intermediate = lora_shrink(x, lora_a)
        out_triton = lora_expand(intermediate, lora_b) * scale

        # Reference
        out_ref = lora_forward_ref(x.float(), lora_a.float(), lora_b.float(), scale).half()

        # fp16 accumulates error across two matmuls, so use slightly looser tolerance
        torch.testing.assert_close(out_triton, out_ref, rtol=5e-2, atol=5e-2)


class TestBatchedLoRA:
    """Tests for batched LoRA operations."""

    def test_batched_lora_indices(self):
        """Test batched LoRA with different adapters per sample."""
        batch_size = 8
        hidden_dim = 256
        lora_rank = 16
        output_dim = 512
        num_loras = 4

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float16)
        lora_a_stacked = torch.randn(num_loras, lora_rank, hidden_dim, device="cuda", dtype=torch.float16)
        lora_b_stacked = torch.randn(num_loras, output_dim, lora_rank, device="cuda", dtype=torch.float16)
        indices = torch.randint(-1, num_loras, (batch_size,), device="cuda")  # -1 = no LoRA

        # Reference implementation
        out_ref = batched_lora_forward_ref(
            x.float(), lora_a_stacked.float(), lora_b_stacked.float(), indices
        ).half()

        # Check output shape
        assert out_ref.shape == (batch_size, output_dim)

        # Check that samples with index -1 have zero output
        for i in range(batch_size):
            if indices[i] == -1:
                assert torch.allclose(out_ref[i], torch.zeros_like(out_ref[i]))


class TestLoRAMath:
    """Tests for LoRA mathematical properties."""

    def test_lora_low_rank_property(self):
        """Test that LoRA produces low-rank updates."""
        hidden_dim = 256
        lora_rank = 16
        output_dim = 512

        lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float32)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float32)

        # Full weight matrix: B @ A
        delta_w = lora_b @ lora_a

        # Check rank using relative threshold based on max singular value
        _, s, _ = torch.svd(delta_w)
        # Use relative threshold: singular values < 1e-5 * max are considered zero
        threshold = s.max() * 1e-5
        effective_rank = (s > threshold).sum().item()
        assert effective_rank == lora_rank, f"Expected rank {lora_rank}, got {effective_rank}"

    def test_lora_scaling(self):
        """Test LoRA scaling property: scale * (x @ A.T @ B.T)."""
        batch_size = 8
        hidden_dim = 256
        lora_rank = 16
        output_dim = 512

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float32)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float32)

        # Different scales
        scale1 = 0.5
        scale2 = 2.0

        out1 = lora_forward_ref(x, lora_a, lora_b, scale1)
        out2 = lora_forward_ref(x, lora_a, lora_b, scale2)

        # Outputs should differ by ratio of scales
        expected_ratio = scale2 / scale1
        torch.testing.assert_close(out2 / out1, torch.full_like(out1, expected_ratio), rtol=1e-5, atol=1e-5)

    def test_lora_additivity(self):
        """Test LoRA additivity with base model."""
        batch_size = 8
        hidden_dim = 256
        output_dim = 512
        lora_rank = 16

        x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        base_weight = torch.randn(output_dim, hidden_dim, device="cuda", dtype=torch.float32)
        lora_a = torch.randn(lora_rank, hidden_dim, device="cuda", dtype=torch.float32)
        lora_b = torch.randn(output_dim, lora_rank, device="cuda", dtype=torch.float32)

        # Base output
        base_out = x @ base_weight.T

        # LoRA output
        lora_out = lora_forward_ref(x, lora_a, lora_b)

        # Combined output: base + LoRA
        combined = base_out + lora_out

        # This is equivalent to: x @ (W + BA).T
        merged_weight = base_weight + lora_b @ lora_a
        merged_out = x @ merged_weight.T

        # Use slightly looser tolerance for numerical stability
        torch.testing.assert_close(combined, merged_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
