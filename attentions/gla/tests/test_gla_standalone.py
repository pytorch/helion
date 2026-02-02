# GLA (Gated Linear Attention) Standalone Tests
# Tests the naive implementation without requiring the full fla package

import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch


# PyTorch reference implementation
def gla_reference(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    gk: torch.Tensor,  # [B, T, H, K] - gate in log space
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch reference implementation of Gated Linear Attention.

    This is an independent implementation for testing purposes.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    q, k, v, gk = map(lambda x: x.float(), [q, k, v, gk])
    scale = K ** -0.5

    # Initialize state
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state.float().clone()

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        # Extract current timestep
        q_t = q[:, t] * scale  # [B, H, K]
        k_t = k[:, t]  # [B, H, K]
        v_t = v[:, t]  # [B, H, V]
        gk_t = gk[:, t].exp()  # [B, H, K]

        # Update state: h = h * gk + k @ v^T
        kv = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, H, K, V]
        h = h * gk_t.unsqueeze(-1) + kv

        # Output: o = q @ h
        o[:, t] = (q_t.unsqueeze(-1) * h).sum(-2)  # [B, H, V]

    if not output_final_state:
        h = None

    return o.to(dtype), h


def get_device():
    """Get the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
    (2, 256, 4, 64, 128),
])
@torch.inference_mode()
def test_naive_recurrent_gla_vs_reference(B: int, T: int, H: int, K: int, V: int):
    """Test naive_recurrent_gla against PyTorch reference."""
    from naive import naive_recurrent_gla

    device = get_device()
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    gk = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1  # Small values for stability
    h0 = torch.randn(B, H, K, V, dtype=dtype, device=device)

    # Run reference
    ref_o, ref_h = gla_reference(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        gk=gk.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run naive implementation
    tri_o, tri_h = naive_recurrent_gla(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        gk=gk.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Compare
    torch.testing.assert_close(tri_o, ref_o, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tri_h, ref_h, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_naive_recurrent_gla_basic(B: int, T: int, H: int, K: int, V: int):
    """Basic test for naive_recurrent_gla without initial state."""
    from naive import naive_recurrent_gla

    device = get_device()
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    gk = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1

    # Run without initial state
    o, final_state = naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=gk,
        initial_state=None,
        output_final_state=True,
    )

    # Basic shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is not all zeros
    assert o.abs().sum() > 0, "Output should not be all zeros"
    assert final_state.abs().sum() > 0, "Final state should not be all zeros"


@pytest.mark.parametrize("B,T,H,K,V", [
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_naive_recurrent_gla_without_final_state(B: int, T: int, H: int, K: int, V: int):
    """Test naive_recurrent_gla without returning final state."""
    from naive import naive_recurrent_gla

    device = get_device()
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    gk = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1

    # Run without final state
    o, final_state = naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=gk,
        initial_state=None,
        output_final_state=False,
    )

    # Shape check
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state is None, "Final state should be None when output_final_state=False"


if __name__ == "__main__":
    # Run quick tests
    print("Testing GLA naive recurrent vs reference...")
    test_naive_recurrent_gla_vs_reference(B=2, T=128, H=4, K=64, V=64)
    print("  Naive recurrent vs reference test passed!")

    print("Testing GLA basic naive recurrent...")
    test_naive_recurrent_gla_basic(B=2, T=128, H=4, K=64, V=64)
    print("  Basic naive recurrent test passed!")

    print("Testing GLA without final state...")
    test_naive_recurrent_gla_without_final_state(B=2, T=128, H=4, K=64, V=64)
    print("  Without final state test passed!")

    print("\nAll GLA standalone tests passed!")
