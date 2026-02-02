# KDA (Key-Differential Attention) Standalone Tests
# Tests the naive implementations and gate functions without requiring the full fla package

import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "common"))

import pytest
import torch
import torch.nn.functional as F

from naive import naive_recurrent_kda, naive_chunk_kda
from gate import naive_kda_gate, naive_kda_lowerbound_gate, fused_kda_gate
from fla_utils import assert_close


def get_device():
    """Get the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.mark.parametrize("B,T,H,D,scale,gate_logit_normalizer", [
    (1, 64, 1, 64, 1, 1),
    (2, 128, 2, 32, 1, 1),
    (2, 256, 4, 64, 0.1, 1),
])
@torch.inference_mode()
def test_naive_chunk_vs_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
):
    """Test that naive_chunk_kda produces the same results as naive_recurrent_kda."""
    device = get_device()
    torch.manual_seed(42)

    dtype = torch.float32  # Use float32 for reference comparison

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device)) / gate_logit_normalizer
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=dtype, device=device)

    # Normalize q and k
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)

    # Run recurrent (reference)
    ref_o, ref_ht = naive_recurrent_kda(
        q=q_norm.clone(),
        k=k_norm.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run chunk implementation
    tri_o, tri_ht = naive_chunk_kda(
        q=q_norm.clone(),
        k=k_norm.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        chunk_size=64,
    )

    assert_close("output", ref_o, tri_o, 0.005)
    assert_close("final_state", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize("B,T,H,D,has_bias,lower_bound", [
    (1, 32, 2, 16, False, None),
    (2, 64, 4, 32, False, None),
    (4, 128, 8, 64, True, None),
    (2, 64, 4, 32, False, -5.0),
    (4, 128, 8, 64, True, -5.0),
])
def test_gate_functions(
    B: int,
    T: int,
    H: int,
    D: int,
    has_bias: bool,
    lower_bound: float | None,
):
    """Test KDA gate functions: naive vs fused implementations."""
    device = get_device()
    torch.manual_seed(42)

    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device) * 10
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device) if has_bias else None

    g = g.requires_grad_(True)
    A_log = A_log.requires_grad_(True)
    if dt_bias is not None:
        dt_bias = dt_bias.requires_grad_(True)

    do = torch.randn_like(g)

    # Run naive gate
    if lower_bound is not None:
        ref = naive_kda_lowerbound_gate(
            g.clone(), A_log.clone(),
            dt_bias.clone() if dt_bias is not None else None,
            lower_bound
        )
    else:
        ref = naive_kda_gate(
            g.clone(), A_log.clone(),
            dt_bias.clone() if dt_bias is not None else None,
        )

    # Run fused gate
    tri = fused_kda_gate(
        g.clone(), A_log.clone(),
        dt_bias.clone() if dt_bias is not None else None,
        lower_bound=lower_bound
    )

    # Forward comparison
    assert_close("gate_output", ref, tri, 1e-4)

    # Backward comparison
    (ref * do).sum().backward(retain_graph=True)
    ref_dg, ref_dA = g.grad.clone(), A_log.grad.clone()
    ref_dbias = dt_bias.grad.clone() if dt_bias is not None else None
    g.grad, A_log.grad = None, None
    if dt_bias is not None:
        dt_bias.grad = None

    (tri * do).sum().backward(retain_graph=True)
    tri_dg, tri_dA = g.grad, A_log.grad
    tri_dbias = dt_bias.grad if dt_bias is not None else None

    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dA", ref_dA, tri_dA, 1e-4)
    if has_bias:
        assert_close("dbias", ref_dbias, tri_dbias, 1e-4)


@pytest.mark.parametrize("B,T,H,D", [
    (1, 64, 2, 32),
    (2, 128, 4, 64),
])
@torch.inference_mode()
def test_naive_recurrent_basic(B: int, T: int, H: int, D: int):
    """Basic test for naive_recurrent_kda without initial state."""
    device = get_device()
    torch.manual_seed(42)

    dtype = torch.float32

    q = torch.rand(B, T, H, D, dtype=dtype, device=device)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype, device=device))
    beta = torch.rand(B, T, H, dtype=dtype, device=device)

    # Normalize q and k
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)

    # Run without initial state
    o, final_state = naive_recurrent_kda(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=None,
        output_final_state=True,
    )

    # Basic shape checks
    assert o.shape == (B, T, H, D), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, D, D), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is not all zeros
    assert o.abs().sum() > 0, "Output should not be all zeros"
    assert final_state.abs().sum() > 0, "Final state should not be all zeros"


@pytest.mark.parametrize("B,T,H,D", [
    (1, 64, 2, 32),
    (2, 128, 4, 64),
])
@torch.inference_mode()
def test_naive_kda_gate_basic(B: int, T: int, H: int, D: int):
    """Basic test for naive_kda_gate."""
    device = get_device()
    torch.manual_seed(42)

    g = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    A_log = torch.log(torch.rand(H, dtype=torch.float32, device=device) + 1)  # Positive values

    # Run gate function
    output = naive_kda_gate(g, A_log, dt_bias=None)

    # Basic shape check
    assert output.shape == (B, T, H, D), f"Output shape mismatch: {output.shape}"

    # The output should be negative (due to -exp(A_log) * softplus(g))
    # softplus is always positive, exp is always positive
    assert (output <= 0).all(), "Gate output should be non-positive"


if __name__ == "__main__":
    # Run quick tests
    print("Testing KDA naive chunk vs recurrent...")
    test_naive_chunk_vs_recurrent(B=2, T=128, H=2, D=32, scale=1.0, gate_logit_normalizer=1)
    print("  Naive chunk vs recurrent test passed!")

    print("Testing KDA gate functions...")
    test_gate_functions(B=2, T=64, H=4, D=32, has_bias=True, lower_bound=None)
    print("  Gate functions test passed!")

    print("Testing KDA gate functions with lower bound...")
    test_gate_functions(B=2, T=64, H=4, D=32, has_bias=True, lower_bound=-5.0)
    print("  Gate functions with lower bound test passed!")

    print("Testing basic naive recurrent...")
    test_naive_recurrent_basic(B=2, T=128, H=4, D=64)
    print("  Basic naive recurrent test passed!")

    print("Testing basic naive gate...")
    test_naive_kda_gate_basic(B=2, T=128, H=4, D=64)
    print("  Basic naive gate test passed!")

    print("\nAll KDA standalone tests passed!")
