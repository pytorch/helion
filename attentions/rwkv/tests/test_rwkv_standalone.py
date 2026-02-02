# RWKV Standalone Tests
# Tests the fused recurrent Triton implementations without requiring the full fla package

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


# RWKV6 Tests
@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_recurrent_rwkv6_basic(B: int, T: int, H: int, K: int, V: int):
    """Basic test for fused_recurrent_rwkv6."""
    from rwkv6.fused_recurrent import fused_recurrent_rwkv6

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    r = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    # w is the log gate, should be negative for stability
    w = -torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.1
    # u is the bonus term
    u = torch.randn(H, K, dtype=dtype, device=device) * 0.1

    # Run without initial state
    o, final_state = fused_recurrent_rwkv6(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        scale=K ** -0.5,
        initial_state=None,
        output_final_state=True,
    )

    # Basic shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is finite
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_state).all(), "Final state contains NaN or inf"


@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_recurrent_rwkv6_with_initial_state(B: int, T: int, H: int, K: int, V: int):
    """Test fused_recurrent_rwkv6 with initial state."""
    from rwkv6.fused_recurrent import fused_recurrent_rwkv6

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    r = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    w = -torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.1
    u = torch.randn(H, K, dtype=dtype, device=device) * 0.1
    h0 = torch.randn(B, H, K, V, dtype=dtype, device=device) * 0.1

    # Run with initial state
    o, final_state = fused_recurrent_rwkv6(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        scale=K ** -0.5,
        initial_state=h0,
        output_final_state=True,
    )

    # Shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is finite
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_state).all(), "Final state contains NaN or inf"


# RWKV7 Tests
@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 2, 32, 32),
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_mul_recurrent_rwkv7_basic(B: int, T: int, H: int, K: int, V: int):
    """Basic test for fused_mul_recurrent_rwkv7."""
    from rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    r = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    w = -torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.1  # log decay
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    kk = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1

    # Run without initial state
    o, final_state = fused_mul_recurrent_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        scale=K ** -0.5,
        initial_state=None,
        output_final_state=True,
    )

    # Basic shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is finite
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_state).all(), "Final state contains NaN or inf"


@requires_cuda
@pytest.mark.parametrize("B,T,H,K,V", [
    (2, 128, 4, 64, 64),
])
@torch.inference_mode()
def test_fused_mul_recurrent_rwkv7_with_initial_state(B: int, T: int, H: int, K: int, V: int):
    """Test fused_mul_recurrent_rwkv7 with initial state."""
    from rwkv7.fused_recurrent import fused_mul_recurrent_rwkv7

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    r = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    w = -torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
    kk = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
    h0 = torch.randn(B, H, K, V, dtype=dtype, device=device) * 0.1

    # Run with initial state
    o, final_state = fused_mul_recurrent_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        scale=K ** -0.5,
        initial_state=h0,
        output_final_state=True,
    )

    # Shape checks
    assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
    assert final_state.shape == (B, H, K, V), f"Final state shape mismatch: {final_state.shape}"

    # Check that output is finite
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_state).all(), "Final state contains NaN or inf"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    # Run quick tests
    print("Testing RWKV6 fused recurrent basic...")
    test_fused_recurrent_rwkv6_basic(B=2, T=128, H=4, K=64, V=64)
    print("  RWKV6 basic test passed!")

    print("Testing RWKV6 fused recurrent with initial state...")
    test_fused_recurrent_rwkv6_with_initial_state(B=2, T=128, H=4, K=64, V=64)
    print("  RWKV6 with initial state test passed!")

    print("Testing RWKV7 fused mul recurrent basic...")
    test_fused_mul_recurrent_rwkv7_basic(B=2, T=128, H=4, K=64, V=64)
    print("  RWKV7 basic test passed!")

    print("Testing RWKV7 fused mul recurrent with initial state...")
    test_fused_mul_recurrent_rwkv7_with_initial_state(B=2, T=128, H=4, K=64, V=64)
    print("  RWKV7 with initial state test passed!")

    print("\nAll RWKV standalone tests passed!")
