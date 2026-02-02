# Gated Delta Rule Standalone Tests
# Tests the Triton fused recurrent implementation without requiring the full fla package

import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch


def get_device():
    """Get the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def requires_cuda(func):
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


# PyTorch reference implementation
def gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch reference implementation of Gated Delta Rule.

    This is an independent implementation for testing purposes.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    q, k, v, beta, g = map(lambda x: x.float(), [q, k, v, beta, g])

    if scale is None:
        scale = K ** -0.5

    # Initialize state
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state.float().clone()

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
    q_scaled = q * scale

    for t in range(T):
        # Extract current timestep
        q_t = q_scaled[:, t]  # [B, H, K]
        k_t = k[:, t]  # [B, H, K]
        v_t = v[:, t]  # [B, H, V]
        beta_t = beta[:, t]  # [B, H]
        g_t = g[:, t]  # [B, H]

        # Apply decay gate
        h = h * g_t[:, :, None, None].exp()

        # Delta rule update
        # v' = v - h @ k
        v_prime = v_t - (h * k_t[:, :, :, None]).sum(-2)
        v_prime = v_prime * beta_t[:, :, None]

        # h = h + outer(k, v')
        h = h + k_t[:, :, :, None] * v_prime[:, :, None, :]

        # Output: o = q @ h
        o[:, t] = (q_t[:, :, :, None] * h).sum(-2)

    if not output_final_state:
        h = None

    return o.to(dtype), h


@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
    (2, 256, 4, 64, 128),
])
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_vs_reference(B: int, T: int, H: int, K: int, V: int):
    """Test fused_recurrent_gated_delta_rule Triton kernel against PyTorch reference."""
    from fused_recurrent import fused_recurrent_gated_delta_rule

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=dtype, device=device))
    # Use negative values for g to prevent exponential growth
    g = -torch.rand(B, T, H, dtype=dtype, device=device) * 0.1
    h0 = torch.randn(B, H, K, V, dtype=dtype, device=device) * 0.1

    # Run reference
    ref_o, ref_h = gated_delta_rule_reference(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run Triton fused recurrent implementation
    tri_o, tri_h = fused_recurrent_gated_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Compare
    torch.testing.assert_close(tri_o, ref_o, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tri_h, ref_h, atol=1e-3, rtol=1e-3)


@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_basic(B: int, T: int, H: int, K: int, V: int):
    """Basic test for fused_recurrent_gated_delta_rule without initial state."""
    from fused_recurrent import fused_recurrent_gated_delta_rule

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=dtype, device=device))
    # Use negative values for g to prevent exponential growth
    g = -torch.rand(B, T, H, dtype=dtype, device=device) * 0.1

    # Run without initial state
    o, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=None,
        output_final_state=True,
    )

    # Basic shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is finite (not NaN or inf)
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_state).all(), "Final state contains NaN or inf"


@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_without_final_state(B: int, T: int, H: int, K: int, V: int):
    """Test fused_recurrent_gated_delta_rule without returning final state."""
    from fused_recurrent import fused_recurrent_gated_delta_rule

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=dtype, device=device))
    # Use negative values for g to prevent exponential growth
    g = -torch.rand(B, T, H, dtype=dtype, device=device) * 0.1

    # Run without final state
    o, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=None,
        output_final_state=False,
    )

    # Shape check
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state is None, "Final state should be None when output_final_state=False"
    assert torch.isfinite(o).all(), "Output contains NaN or inf"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    # Run quick tests
    print("Testing Gated Delta Rule fused recurrent vs reference...")
    test_fused_recurrent_gated_delta_rule_vs_reference(B=2, T=128, H=4, K=64, V=64)
    print("  Fused recurrent vs reference test passed!")

    print("Testing Gated Delta Rule basic fused recurrent...")
    test_fused_recurrent_gated_delta_rule_basic(B=2, T=128, H=4, K=64, V=64)
    print("  Basic fused recurrent test passed!")

    print("Testing Gated Delta Rule without final state...")
    test_fused_recurrent_gated_delta_rule_without_final_state(B=2, T=128, H=4, K=64, V=64)
    print("  Without final state test passed!")

    print("\nAll Gated Delta Rule standalone tests passed!")
