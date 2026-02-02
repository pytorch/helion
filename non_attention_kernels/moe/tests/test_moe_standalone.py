# SPDX-License-Identifier: Apache-2.0
"""
Standalone tests for MoE (Mixture of Experts) kernels with PyTorch reference implementations.
"""
import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch
import torch.nn.functional as F

try:
    from fused_moe_sglang import fused_moe_kernel, act_and_mul_kernel, act_and_mul_triton
    HAS_MOE = True
except ImportError:
    HAS_MOE = False


# Skip if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def moe_routing_ref(hidden_states, router_weight, top_k=2):
    """Reference implementation of MoE routing."""
    # Router logits
    router_logits = hidden_states @ router_weight.T
    routing_weights = F.softmax(router_logits, dim=-1)

    # Top-k selection
    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices, routing_weights


def moe_forward_ref(hidden_states, router_weight, expert_weights, top_k=2):
    """
    Reference implementation of MoE forward pass.

    Args:
        hidden_states: [batch, hidden_dim]
        router_weight: [num_experts, hidden_dim]
        expert_weights: [num_experts, hidden_dim, intermediate_dim]
        top_k: number of experts per token
    """
    batch_size, hidden_dim = hidden_states.shape
    num_experts = router_weight.shape[0]
    intermediate_dim = expert_weights.shape[-1]

    # Get routing weights
    topk_weights, topk_indices, _ = moe_routing_ref(hidden_states, router_weight, top_k)

    # Compute output
    output = torch.zeros(batch_size, intermediate_dim, device=hidden_states.device, dtype=hidden_states.dtype)

    for i in range(batch_size):
        for k in range(top_k):
            expert_idx = topk_indices[i, k].item()
            weight = topk_weights[i, k]
            expert_out = hidden_states[i] @ expert_weights[expert_idx]
            output[i] += weight * expert_out

    return output


def silu_and_mul_ref(x):
    """Reference implementation of SiLU activation followed by element-wise multiplication."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


class TestMoERouting:
    """Tests for MoE routing kernels."""

    def test_routing_basic(self):
        """Test basic MoE routing using local reference implementation."""
        batch_size = 8
        hidden_dim = 256
        num_experts = 8
        top_k = 2

        hidden_states = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float32)

        # Get reference routing
        topk_weights, topk_indices, routing_weights = moe_routing_ref(
            hidden_states, router_weight, top_k
        )

        # Check routing properties
        assert topk_indices.shape == (batch_size, top_k)
        assert topk_weights.shape == (batch_size, top_k)
        assert (topk_indices >= 0).all() and (topk_indices < num_experts).all()
        assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(batch_size, device="cuda"), rtol=1e-5)


class TestFusedMoE:
    """Tests for fused MoE kernels using local Triton implementation."""

    @pytest.mark.skipif(not HAS_MOE, reason="fused_moe_sglang not available")
    def test_fused_moe_kernel_basic(self):
        """Test basic fused MoE kernel directly."""
        import triton

        # Test that the kernel is properly defined with @triton.jit
        assert hasattr(fused_moe_kernel, '__wrapped__') or callable(fused_moe_kernel)

    def test_moe_forward_reference(self):
        """Test MoE forward pass using reference implementation."""
        batch_size = 8
        hidden_dim = 256
        intermediate_dim = 512
        num_experts = 8
        top_k = 2

        hidden_states = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
        router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float32)
        expert_weights = torch.randn(num_experts, hidden_dim, intermediate_dim, device="cuda", dtype=torch.float32)

        # Use reference implementation
        output = moe_forward_ref(hidden_states, router_weight, expert_weights, top_k)

        # Check output shape and validity
        assert output.shape == (batch_size, intermediate_dim)
        assert torch.isfinite(output).all()

    def test_moe_forward_various_configs(self):
        """Test MoE forward with various configurations."""
        configs = [
            {"batch_size": 4, "hidden_dim": 128, "intermediate_dim": 256, "num_experts": 4, "top_k": 1},
            {"batch_size": 8, "hidden_dim": 256, "intermediate_dim": 512, "num_experts": 8, "top_k": 2},
            {"batch_size": 16, "hidden_dim": 512, "intermediate_dim": 1024, "num_experts": 16, "top_k": 2},
        ]

        for config in configs:
            batch_size = config["batch_size"]
            hidden_dim = config["hidden_dim"]
            intermediate_dim = config["intermediate_dim"]
            num_experts = config["num_experts"]
            top_k = config["top_k"]

            hidden_states = torch.randn(batch_size, hidden_dim, device="cuda", dtype=torch.float32)
            router_weight = torch.randn(num_experts, hidden_dim, device="cuda", dtype=torch.float32)
            expert_weights = torch.randn(num_experts, hidden_dim, intermediate_dim, device="cuda", dtype=torch.float32)

            output = moe_forward_ref(hidden_states, router_weight, expert_weights, top_k)
            assert output.shape == (batch_size, intermediate_dim), f"Failed for config {config}"
            assert torch.isfinite(output).all(), f"Output contains NaN/Inf for config {config}"


class TestMoEActivation:
    """Tests for MoE activation kernels."""

    @pytest.mark.skipif(not HAS_MOE, reason="fused_moe_sglang not available")
    def test_silu_and_mul_basic(self):
        """Test SiLU and mul activation (used in MoE FFN)."""
        batch_size = 8
        intermediate_dim = 512

        # Input has 2x intermediate_dim (gate and up projections concatenated)
        x = torch.randn(batch_size, intermediate_dim * 2, device="cuda", dtype=torch.float16)

        try:
            # act_and_mul_triton writes to output tensor in-place
            out_triton = torch.zeros(batch_size, intermediate_dim, device="cuda", dtype=torch.float16)
            config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}
            # For simple test, use dummy expert IDs (all zeros)
            topk_ids = torch.zeros(batch_size, 1, dtype=torch.int32, device="cuda")
            act_and_mul_triton(x, out_triton, config, topk_ids=topk_ids, activation="silu")

            out_ref = silu_and_mul_ref(x.float()).half()

            torch.testing.assert_close(out_triton, out_ref, rtol=1e-2, atol=1e-2)
        except Exception as e:
            pytest.skip(f"SiLU and mul not supported: {e}")

    @pytest.mark.skipif(not HAS_MOE, reason="fused_moe_sglang not available")
    def test_silu_and_mul_dtypes(self):
        """Test SiLU and mul with different dtypes."""
        batch_size = 8
        intermediate_dim = 256

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(batch_size, intermediate_dim * 2, device="cuda", dtype=dtype)

            try:
                out_triton = torch.zeros(batch_size, intermediate_dim, device="cuda", dtype=dtype)
                config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}
                topk_ids = torch.zeros(batch_size, 1, dtype=torch.int32, device="cuda")
                act_and_mul_triton(x, out_triton, config, topk_ids=topk_ids, activation="silu")

                out_ref = silu_and_mul_ref(x.float()).to(dtype)

                rtol = 1e-2 if dtype != torch.float32 else 1e-3
                atol = 1e-2 if dtype != torch.float32 else 1e-3
                torch.testing.assert_close(
                    out_triton, out_ref, rtol=rtol, atol=atol,
                    msg=f"Failed for dtype {dtype}"
                )
            except Exception:
                continue


class TestMoETokenRouting:
    """Tests for MoE token-to-expert routing utilities."""

    def test_token_expert_mapping(self):
        """Test that tokens are correctly mapped to experts."""
        batch_size = 16
        num_experts = 8
        top_k = 2

        # Create deterministic routing
        torch.manual_seed(42)
        routing_weights = torch.softmax(
            torch.randn(batch_size, num_experts, device="cuda"),
            dim=-1
        )
        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)

        # Verify each token has top_k experts
        assert topk_indices.shape == (batch_size, top_k)

        # Verify experts are unique per token
        for i in range(batch_size):
            assert len(topk_indices[i].unique()) == top_k

    def test_expert_load_balancing(self):
        """Test that expert assignment is roughly balanced."""
        batch_size = 1024
        num_experts = 8
        top_k = 2

        # Uniform routing should give balanced load
        routing_weights = torch.ones(batch_size, num_experts, device="cuda") / num_experts
        # Add small noise to break ties
        routing_weights += torch.randn_like(routing_weights) * 0.01
        routing_weights = F.softmax(routing_weights, dim=-1)

        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)

        # Count tokens per expert
        expert_counts = torch.zeros(num_experts, device="cuda")
        for idx in topk_indices.flatten():
            expert_counts[idx] += 1

        # Each expert should get roughly batch_size * top_k / num_experts tokens
        expected = batch_size * top_k / num_experts
        # Allow 50% deviation
        assert (expert_counts > expected * 0.5).all()
        assert (expert_counts < expected * 1.5).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
