# Mamba Standalone Tests
# Tests the Mamba SSM Triton implementations without requiring the full mamba_ssm package

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


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base if base > 0 else 0


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


@requires_cuda
@pytest.mark.parametrize("batch,nheads,dim,dstate,ngroups", [
    (2, 4, 64, 16, 2),
    (1, 8, 128, 32, 4),
])
@torch.inference_mode()
def test_selective_state_update(batch: int, nheads: int, dim: int, dstate: int, ngroups: int):
    """Test the selective state update kernel."""
    from selective_state_update import selective_state_update, selective_state_update_ref

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create inputs
    state = torch.randn(batch, nheads, dim, dstate, dtype=dtype, device=device) * 0.1
    x = torch.randn(batch, nheads, dim, dtype=dtype, device=device) * 0.1
    dt = torch.randn(batch, nheads, dim, dtype=dtype, device=device).abs() * 0.1
    A = -torch.rand(nheads, dim, dstate, dtype=dtype, device=device) * 0.1
    B = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device) * 0.1
    C = torch.randn(batch, ngroups, dstate, dtype=dtype, device=device) * 0.1
    D = torch.randn(nheads, dim, dtype=dtype, device=device) * 0.1
    z = torch.randn(batch, nheads, dim, dtype=dtype, device=device) * 0.1
    dt_bias = torch.randn(nheads, dim, dtype=dtype, device=device) * 0.1

    # Clone state for reference
    state_ref = state.clone()
    state_tri = state.clone()

    # Run reference
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    # Run Triton kernel
    out_tri = selective_state_update(
        state_tri, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    # Check output
    assert out_tri.shape == out_ref.shape, f"Output shape mismatch: {out_tri.shape} vs {out_ref.shape}"
    assert torch.isfinite(out_tri).all(), "Output contains NaN or inf"
    assert_close("selective_state_update output", out_ref, out_tri, 0.01)

    # Check state update
    assert_close("selective_state_update state", state_ref, state_tri, 0.01)


@requires_cuda
@pytest.mark.parametrize("batch,nchunks,nheads,dim", [
    (2, 4, 4, 64),
    (1, 8, 8, 128),
])
@torch.inference_mode()
def test_state_passing(batch: int, nchunks: int, nheads: int, dim: int):
    """Test the state passing kernel."""
    from ssd_state_passing import state_passing, state_passing_ref

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create inputs
    states = torch.randn(batch, nchunks, nheads, dim, dtype=dtype, device=device) * 0.1
    dA_chunk_cumsum = torch.randn(batch, nheads, nchunks, dtype=dtype, device=device) * 0.1
    initial_states = torch.randn(batch, nheads, dim, dtype=dtype, device=device) * 0.1

    # Run reference
    out_ref, final_states_ref = state_passing_ref(states, dA_chunk_cumsum, initial_states)

    # Run Triton kernel
    out_tri, final_states_tri = state_passing(states, dA_chunk_cumsum, initial_states)

    # Check outputs
    assert out_tri.shape == out_ref.shape, f"Output shape mismatch: {out_tri.shape} vs {out_ref.shape}"
    assert final_states_tri.shape == final_states_ref.shape, f"Final states shape mismatch"
    assert torch.isfinite(out_tri).all(), "Output contains NaN or inf"
    assert torch.isfinite(final_states_tri).all(), "Final states contains NaN or inf"
    assert_close("state_passing output", out_ref, out_tri, 0.01)
    assert_close("state_passing final_states", final_states_ref, final_states_tri, 0.01)


@requires_cuda
@pytest.mark.parametrize("batch,seqlen,k,ngroups,chunk_size", [
    (2, 128, 64, 1, 32),
    (1, 256, 32, 4, 64),
])
@torch.inference_mode()
def test_bmm_chunk_fwd(batch: int, seqlen: int, k: int, ngroups: int, chunk_size: int):
    """Test the BMM chunk forward kernel."""
    from ssd_bmm import _bmm_chunk_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create inputs
    if ngroups == 1:
        a = torch.randn(batch, seqlen, k, dtype=dtype, device=device) * 0.1
        b = torch.randn(batch, seqlen, k, dtype=dtype, device=device) * 0.1
    else:
        a = torch.randn(batch, seqlen, ngroups, k, dtype=dtype, device=device) * 0.1
        b = torch.randn(batch, seqlen, ngroups, k, dtype=dtype, device=device) * 0.1

    # Run kernel
    out = _bmm_chunk_fwd(a, b, chunk_size)

    # Basic checks
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    if ngroups == 1:
        expected_shape = (batch, nchunks, chunk_size, chunk_size)
    else:
        expected_shape = (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert out.shape == expected_shape, f"Output shape mismatch: {out.shape} vs {expected_shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or inf"


@requires_cuda
@pytest.mark.parametrize("M,N", [
    (128, 64),
    (256, 128),
])
@torch.inference_mode()
def test_swiglu_fwd(M: int, N: int):
    """Test the SwiGLU forward kernel."""
    from k_activations import _swiglu_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create input (x and y concatenated)
    xy = torch.randn(M, N * 2, dtype=dtype, device=device) * 0.1

    # Run kernel
    out = _swiglu_fwd(xy)

    # Check output
    assert out.shape == (M, N), f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or inf"

    # Verify against reference
    x, y = xy.chunk(2, dim=-1)
    out_ref = x * torch.sigmoid(x) * y
    assert_close("swiglu output", out_ref, out, 0.001)


@requires_cuda
@pytest.mark.parametrize("M,N", [
    (128, 64),
    (256, 128),
])
@torch.inference_mode()
def test_layer_norm_fwd(M: int, N: int):
    """Test the layer norm forward kernel."""
    from layernorm_gated import _layer_norm_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create inputs
    x = torch.randn(M, N, dtype=dtype, device=device) * 0.1
    x = x.contiguous()
    weight = torch.ones(N, dtype=dtype, device=device)
    bias = torch.zeros(N, dtype=dtype, device=device)
    eps = 1e-5

    # Run kernel
    out, mean, rstd = _layer_norm_fwd(x, weight, bias, eps)

    # Check output
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or inf"

    # Verify against PyTorch reference
    out_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    assert_close("layer_norm output", out_ref, out, 0.001)


@requires_cuda
@pytest.mark.parametrize("M,N", [
    (128, 64),
    (256, 128),
])
@torch.inference_mode()
def test_rms_norm_fwd(M: int, N: int):
    """Test the RMS norm forward kernel."""
    from layernorm_gated import _layer_norm_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float32

    # Create inputs
    x = torch.randn(M, N, dtype=dtype, device=device) * 0.1
    x = x.contiguous()
    weight = torch.ones(N, dtype=dtype, device=device)
    eps = 1e-5

    # Run kernel (is_rms_norm=True)
    out, _, rstd = _layer_norm_fwd(x, weight, None, eps, is_rms_norm=True)

    # Check output
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or inf"

    # Verify against reference
    rstd_ref = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    out_ref = x * rstd_ref * weight
    assert_close("rms_norm output", out_ref, out, 0.001)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    # Run quick tests
    print("Testing selective state update...")
    test_selective_state_update(batch=2, nheads=4, dim=64, dstate=16, ngroups=2)
    print("  Selective state update test passed!")

    print("Testing state passing...")
    test_state_passing(batch=2, nchunks=4, nheads=4, dim=64)
    print("  State passing test passed!")

    print("Testing BMM chunk forward...")
    test_bmm_chunk_fwd(batch=2, seqlen=128, k=64, ngroups=1, chunk_size=32)
    print("  BMM chunk forward test passed!")

    print("Testing SwiGLU forward...")
    test_swiglu_fwd(M=128, N=64)
    print("  SwiGLU forward test passed!")

    print("Testing layer norm forward...")
    test_layer_norm_fwd(M=128, N=64)
    print("  Layer norm forward test passed!")

    print("Testing RMS norm forward...")
    test_rms_norm_fwd(M=128, N=64)
    print("  RMS norm forward test passed!")

    print("\nAll Mamba standalone tests passed!")
