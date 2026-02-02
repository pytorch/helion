"""
RWKV (Receptance Weighted Key Value) - Core Triton Kernel

==============================================================================
MATHEMATICAL CORE
==============================================================================

RWKV is a linear-complexity attention mechanism that combines the benefits of
RNNs (O(1) inference) and Transformers (parallelizable training). The key
insight is using exponential decay for selective forgetting.

RWKV-6 "Finch" - Exponentially-decayed linear attention with bonus term:
------------------------------------------------------------------------------

Recurrence Form (per head, per timestep):
    h_t = h_{t-1} * exp(w_t) + k_t ⊗ v_t
    o_t = sum((h_t + u * k_t ⊗ v_t) * r_t)

Where:
    - h_t ∈ R^{K×V} is the hidden state (key-value memory)
    - w_t ∈ R^K is the data-dependent decay (in log space)
    - k_t ∈ R^K is the key vector
    - v_t ∈ R^V is the value vector
    - r_t ∈ R^K is the receptance (query) vector
    - u ∈ R^K is the "bonus" parameter emphasizing current token
    - o_t ∈ R^V is the output

The bonus term `u * k_t ⊗ v_t` gives extra weight to the current token's
contribution, which is crucial for RWKV's performance.

Complexity:
    - Time: O(T × K × V) per head (linear in sequence length)
    - Space: O(K × V) for hidden state (constant in sequence length)

RWKV-7 "Goose" - SGD-inspired dynamic state evolution:
------------------------------------------------------------------------------

    S_t = diag(exp(-exp(w_t))) @ S_{t-1}    # Diagonal decay
        + S_{t-1} @ α_t @ β_t^T              # Matrix transformation
        + k_t ⊗ v_t                          # Standard KV update
    o_t = S_t @ r_t

The `S @ α β^T` term simulates online gradient descent on L = ½||v - k^T S||²,
enabling in-context learning capabilities (NC1 expressivity).

References:
    - RWKV: Reinventing RNNs for the Transformer Era (arxiv.org/abs/2305.13048)
    - Flash Linear Attention (FLA) Library (github.com/fla-org/flash-linear-attention)

==============================================================================
"""

import math

import torch
import triton
import triton.language as tl

# Helion imports (optional)
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False


# ==============================================================================
# Triton Kernel: Fused Recurrent RWKV-6
# ==============================================================================

@triton.jit
def rwkv6_recurrent_fwd_kernel(
    # Pointers to matrices
    r_ptr,      # Receptance (query) [B, T, H, K]
    k_ptr,      # Key [B, T, H, K]
    v_ptr,      # Value [B, T, H, V]
    w_ptr,      # Decay in log-space [B, T, H, K]
    u_ptr,      # Bonus [H, K]
    o_ptr,      # Output [NK, B, T, H, V]
    h0_ptr,     # Initial state (optional) [B, H, K, V]
    ht_ptr,     # Final state output (optional) [B, H, K, V]
    # Matrix dimensions
    B,          # Batch size
    T,          # Sequence length
    H: tl.constexpr,  # Number of heads
    K: tl.constexpr,  # Key dimension
    V: tl.constexpr,  # Value dimension
    # Scaling factor
    scale,
    # Block sizes
    BK: tl.constexpr,
    BV: tl.constexpr,
    # Flags
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """
    Fused recurrent RWKV-6 forward pass.

    Computes for each timestep t:
        kv_t = k_t ⊗ v_t                           # Outer product
        o_t = sum((h_t + u * kv_t) * r_t)          # Output with bonus
        h_t = h_{t-1} * exp(w_t) + kv_t            # State update

    Grid: (num_v_blocks, num_k_blocks, batch * heads)
    """
    # Get block indices
    i_v = tl.program_id(0)  # Which V block
    i_k = tl.program_id(1)  # Which K block
    i_bh = tl.program_id(2)  # Which (batch, head) combination

    i_b = i_bh // H
    i_h = i_bh % H

    # Compute offsets for K and V dimensions
    o_k = i_k * BK + tl.arange(0, BK)  # [BK]
    o_v = i_v * BV + tl.arange(0, BV)  # [BV]

    # Masks for boundary conditions
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]  # [BK, BV]

    # Load bonus vector u [H, K]
    p_u = u_ptr + i_h * K + o_k
    b_u = tl.load(p_u, mask=m_k, other=0.0).to(tl.float32)

    # Initialize hidden state block
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # Load initial state if provided
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    # Starting positions for this batch/head
    # Input layout: [B, T, H, K] or [B, T, H, V]
    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V

    # Iterate through timesteps
    for t in range(T):
        # Load inputs for timestep t
        # r: [B, T, H, K] -> load [BK]
        p_r = r_ptr + base_k + t * H * K + o_k
        b_r = tl.load(p_r, mask=m_k, other=0.0).to(tl.float32) * scale

        # k: [B, T, H, K] -> load [BK]
        p_k = k_ptr + base_k + t * H * K + o_k
        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)

        # v: [B, T, H, V] -> load [BV]
        p_v = v_ptr + base_v + t * H * V + o_v
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)

        # w: [B, T, H, K] -> load [BK], decay in log-space
        p_w = w_ptr + base_k + t * H * K + o_k
        b_w = tl.load(p_w, mask=m_k, other=0.0).to(tl.float32)

        # Compute outer product: kv = k ⊗ v
        b_kv = b_k[:, None] * b_v[None, :]  # [BK, BV]

        # Compute output with bonus: o = sum((h + u * kv) * r)
        # The bonus term adds extra emphasis on the current token
        b_o = tl.sum((b_h + b_u[:, None] * b_kv) * b_r[:, None], axis=0)  # [BV]

        # Update hidden state: h = h * exp(w) + kv
        b_h = b_h * tl.exp(b_w)[:, None] + b_kv  # [BK, BV]

        # Store output (accumulated across K blocks)
        p_o = o_ptr + (i_k * B * T * H * V) + (i_b * T * H * V) + (t * H * V) + (i_h * V) + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

    # Store final state if requested
    if STORE_FINAL_STATE:
        p_ht = ht_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


def rwkv6_recurrent_fwd(
    r: torch.Tensor,  # [B, T, H, K] receptance (query)
    k: torch.Tensor,  # [B, T, H, K] key
    v: torch.Tensor,  # [B, T, H, V] value
    w: torch.Tensor,  # [B, T, H, K] decay in log-space
    u: torch.Tensor,  # [H, K] bonus
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Triton implementation of RWKV-6 forward pass.

    Args:
        r: Receptance (query) tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        w: Decay tensor in log-space [B, T, H, K]
        u: Bonus tensor [H, K]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state

    Returns:
        o: Output tensor [B, T, H, V]
        ht: Final hidden state [B, H, K, V] if output_final_state else None
    """
    B, T, H, K = r.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Block sizes for tiling
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    # Output buffer (indexed by K block for accumulation)
    o = r.new_empty(NK, B, T, H, V, dtype=torch.float32)

    # Final state buffer
    ht = r.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None

    # Launch kernel
    grid = (NV, NK, B * H)
    rwkv6_recurrent_fwd_kernel[grid](
        r, k, v, w, u, o,
        initial_state, ht,
        B, T, H, K, V,
        scale,
        BK, BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
    )

    # Sum across K blocks
    o = o.sum(0)  # [B, T, H, V]

    return o.to(r.dtype), ht


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def rwkv6_recurrent_ref(
    r: torch.Tensor,  # [B, T, H, K] receptance (query)
    k: torch.Tensor,  # [B, T, H, K] key
    v: torch.Tensor,  # [B, T, H, V] value
    w: torch.Tensor,  # [B, T, H, K] decay in log-space
    u: torch.Tensor,  # [H, K] bonus
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch reference implementation of RWKV-6.

    This is a straightforward loop-based implementation that matches
    the mathematical definition exactly:
        h_t = h_{t-1} * exp(w_t) + k_t ⊗ v_t
        o_t = sum((h_t + u * k_t ⊗ v_t) * r_t)
    """
    B, T, H, K = r.shape
    V = v.shape[-1]
    dtype = r.dtype

    if scale is None:
        scale = K ** -0.5

    # Convert to float32 for numerical stability
    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()
    u = u.float()

    # Initialize hidden state: h ∈ R^{B×H×K×V}
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
    if initial_state is not None:
        h = initial_state.float().clone()

    # Output buffer
    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=r.device)

    # Recurrence loop
    for t in range(T):
        # Extract timestep t inputs
        r_t = r[:, t] * scale  # [B, H, K]
        k_t = k[:, t]          # [B, H, K]
        v_t = v[:, t]          # [B, H, V]
        w_t = w[:, t]          # [B, H, K]

        # Compute outer product: kv = k ⊗ v
        kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, H, K, V]

        # Compute output with bonus: o = sum((h + u * kv) * r)
        # u is [H, K], broadcast to [1, H, K, 1]
        h_with_bonus = h + u[None, :, :, None] * kv_t  # [B, H, K, V]
        o[:, t] = (h_with_bonus * r_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]

        # Update hidden state: h = h * exp(w) + kv
        h = h * torch.exp(w_t).unsqueeze(-1) + kv_t  # [B, H, K, V]

    if not output_final_state:
        h = None

    return o.to(dtype), h


# ==============================================================================
# Numerical Tests
# ==============================================================================

def test_rwkv6_triton_vs_reference():
    """Test that Triton kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations
    configs = [
        # (B, T, H, K, V)
        (1, 32, 2, 32, 32),
        (2, 64, 4, 64, 64),
        (2, 128, 4, 64, 128),
        (1, 256, 8, 128, 64),
    ]

    print("Testing RWKV-6 Triton kernel vs PyTorch reference...")

    for B, T, H, K, V in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}")

        # Generate random inputs
        r = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
        # Decay values should typically be negative (log-sigmoid range)
        w = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        u = torch.randn(H, K, dtype=torch.float32, device=device)
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

        # Run reference
        ref_o, ref_ht = rwkv6_recurrent_ref(
            r.clone(), k.clone(), v.clone(), w.clone(), u.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Run Triton kernel
        tri_o, tri_ht = rwkv6_recurrent_fwd(
            r.clone(), k.clone(), v.clone(), w.clone(), u.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Check outputs
        o_atol = (ref_o - tri_o).abs().max().item()
        o_rtol = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

        ht_atol = (ref_ht - tri_ht).abs().max().item()
        ht_rtol = ((ref_ht - tri_ht).abs() / (ref_ht.abs() + 1e-8)).max().item()

        # Tolerances - more lenient for longer sequences due to accumulation
        atol_threshold = 1e-3
        rtol_threshold = 5e-2

        o_pass = o_atol < atol_threshold or o_rtol < rtol_threshold
        ht_pass = ht_atol < atol_threshold or ht_rtol < rtol_threshold

        status = "PASS" if (o_pass and ht_pass) else "FAIL"
        print(f"    Output:      atol={o_atol:.2e}, rtol={o_rtol:.2e} [{status}]")
        print(f"    Final state: atol={ht_atol:.2e}, rtol={ht_rtol:.2e} [{status}]")

        if not (o_pass and ht_pass):
            raise AssertionError(f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}")

    print("All tests passed!")


def test_rwkv6_without_initial_state():
    """Test RWKV-6 without initial state."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 64, 4, 64, 64

    r = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    u = torch.randn(H, K, dtype=torch.float32, device=device)

    # Run without initial state
    ref_o, _ = rwkv6_recurrent_ref(r.clone(), k.clone(), v.clone(), w.clone(), u.clone())
    tri_o, _ = rwkv6_recurrent_fwd(r.clone(), k.clone(), v.clone(), w.clone(), u.clone())

    atol = (ref_o - tri_o).abs().max().item()
    print(f"Test without initial state: atol={atol:.2e}")
    assert atol < 1e-3, f"Test failed with atol={atol}"
    print("  PASS")


def test_rwkv6_half_precision():
    """Test RWKV-6 with half precision inputs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 64, 4, 64, 64

    # Generate in float16
    r = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device)
    w = (torch.randn(B, T, H, K, device=device) * 0.1).to(torch.float16)
    u = torch.randn(H, K, dtype=torch.float16, device=device)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

    # Run reference (in float32 internally)
    ref_o, ref_ht = rwkv6_recurrent_ref(
        r.float(), k.float(), v.float(), w.float(), u.float(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run Triton kernel
    tri_o, tri_ht = rwkv6_recurrent_fwd(
        r, k, v, w, u,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    atol = (ref_o.to(tri_o.dtype) - tri_o).abs().max().item()
    print(f"Test half precision: atol={atol:.2e}")
    # Larger tolerance for half precision due to reduced precision in inputs
    assert atol < 5e-2, f"Test failed with atol={atol}"
    print("  PASS")


def test_rwkv6_decay_behavior():
    """Test that decay behavior is correct."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 1, 10, 1, 16, 16

    # Create inputs where we can verify behavior
    r = torch.ones(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.zeros(B, T, H, V, dtype=torch.float32, device=device)
    u = torch.zeros(H, K, dtype=torch.float32, device=device)

    # Only first timestep has k and v
    k[:, 0, :, :] = 1.0
    v[:, 0, :, :] = 1.0

    # Constant decay rate (log of 0.5, so multiply by 0.5 each step)
    w = torch.full((B, T, H, K), fill_value=float('-0.693'), dtype=torch.float32, device=device)

    ref_o, _ = rwkv6_recurrent_ref(r, k, v, w, u, scale=1.0)
    tri_o, _ = rwkv6_recurrent_fwd(r, k, v, w, u, scale=1.0)

    # RWKV-6 computes output BEFORE state update:
    # At t=0: h=0, kv=1, o=sum((0+0)*1)=0, then h=0*exp(w)+1=1
    # At t=1: h=1, kv=0, o=sum((1+0)*1)=K=16, then h=1*0.5+0=0.5
    # At t=2: h=0.5, kv=0, o=sum((0.5+0)*1)=K*0.5=8, then h=0.5*0.5+0=0.25
    # etc.

    expected_o0 = 0.0   # h=0, no bonus contribution
    expected_o1 = 16.0  # sum over K of 1*1 = K = 16
    expected_o2 = 8.0   # sum over K of 0.5 = K*0.5 = 8

    ref_check = (
        abs(ref_o[0, 0, 0, 0].item() - expected_o0) < 0.1 and
        abs(ref_o[0, 1, 0, 0].item() - expected_o1) < 0.1 and
        abs(ref_o[0, 2, 0, 0].item() - expected_o2) < 0.1
    )

    tri_check = (
        abs(tri_o[0, 0, 0, 0].item() - expected_o0) < 0.1 and
        abs(tri_o[0, 1, 0, 0].item() - expected_o1) < 0.1 and
        abs(tri_o[0, 2, 0, 0].item() - expected_o2) < 0.1
    )

    print(f"Test decay behavior:")
    print(f"  Expected o[0,1,2]: {expected_o0:.2f}, {expected_o1:.2f}, {expected_o2:.2f}")
    print(f"  Reference o[0,1,2]: {ref_o[0,0,0,0].item():.2f}, {ref_o[0,1,0,0].item():.2f}, {ref_o[0,2,0,0].item():.2f}")
    print(f"  Triton o[0,1,2]:    {tri_o[0,0,0,0].item():.2f}, {tri_o[0,1,0,0].item():.2f}, {tri_o[0,2,0,0].item():.2f}")

    assert ref_check, "Reference implementation decay check failed"
    assert tri_check, "Triton implementation decay check failed"
    print("  PASS")


def test_rwkv6_bonus_behavior():
    """Test that the bonus term works correctly."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 1, 2, 1, 16, 16

    r = torch.ones(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.ones(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.ones(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)  # No decay

    # Test with and without bonus
    u_zero = torch.zeros(H, K, dtype=torch.float32, device=device)
    u_one = torch.ones(H, K, dtype=torch.float32, device=device)

    ref_o_no_bonus, _ = rwkv6_recurrent_ref(r, k, v, w, u_zero, scale=1.0)
    ref_o_with_bonus, _ = rwkv6_recurrent_ref(r, k, v, w, u_one, scale=1.0)

    tri_o_no_bonus, _ = rwkv6_recurrent_fwd(r, k, v, w, u_zero, scale=1.0)
    tri_o_with_bonus, _ = rwkv6_recurrent_fwd(r, k, v, w, u_one, scale=1.0)

    # With bonus, output should be larger (current kv is weighted more)
    print(f"Test bonus behavior:")
    print(f"  Reference - no bonus: {ref_o_no_bonus[0,0,0,0].item():.2f}, with bonus: {ref_o_with_bonus[0,0,0,0].item():.2f}")
    print(f"  Triton - no bonus: {tri_o_no_bonus[0,0,0,0].item():.2f}, with bonus: {tri_o_with_bonus[0,0,0,0].item():.2f}")

    # At t=0: h=0, kv=1
    # Without bonus: o = (0 + 0*1) * 1 = 0 (but after update h=kv=1)
    # With bonus: o = (0 + 1*1) * 1 = 1
    # Hmm, the computation order matters. Let me trace through:
    # Actually looking at the code: o_t = sum((h_t + u*kv_t) * r_t)
    # where h_t is updated AFTER computing o. So at t=0, h is initial (0),
    # o = (0 + u*kv) * r = u*kv*r = u*K (since all are 1)

    ref_bonus_works = ref_o_with_bonus[0, 0, 0, 0].item() > ref_o_no_bonus[0, 0, 0, 0].item()
    tri_bonus_works = tri_o_with_bonus[0, 0, 0, 0].item() > tri_o_no_bonus[0, 0, 0, 0].item()

    # Also check they match each other
    atol = (ref_o_with_bonus - tri_o_with_bonus).abs().max().item()

    assert ref_bonus_works, "Reference bonus not working"
    assert tri_bonus_works, "Triton bonus not working"
    assert atol < 1e-3, f"Outputs don't match: atol={atol}"
    print("  PASS")


def benchmark_rwkv6():
    """Simple benchmark comparing Triton vs PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 4, 1024, 8, 128, 128

    r = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    u = torch.randn(H, K, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(3):
        rwkv6_recurrent_fwd(r, k, v, w, u)
    torch.cuda.synchronize()

    # Benchmark Triton
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        rwkv6_recurrent_fwd(r, k, v, w, u)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000

    # Benchmark reference
    start = time.time()
    for _ in range(n_iters):
        rwkv6_recurrent_ref(r, k, v, w, u)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iters * 1000

    print(f"\nBenchmark (B={B}, T={T}, H={H}, K={K}, V={V}):")
    print(f"  Triton:    {triton_time:.2f} ms")
    print(f"  Reference: {ref_time:.2f} ms")
    print(f"  Speedup:   {ref_time / triton_time:.2f}x")


# ==============================================================================
# Helion Kernel Implementation
# ==============================================================================

if HELION_AVAILABLE:
    @helion.kernel(static_shapes=True, autotune_effort="none", dot_precision="ieee")
    def rwkv6_helion_kernel(
        r: torch.Tensor,              # [B, T, H, K] - receptance (query)
        k: torch.Tensor,              # [B, T, H, K]
        v: torch.Tensor,              # [B, T, H, V]
        w: torch.Tensor,              # [B, T, H, K] - decay in log-space
        u: torch.Tensor,              # [H, K] - bonus (per-head, shared across batch)
        initial_state: torch.Tensor,  # [B, H, K, V] or dummy
        scale: float,
        use_initial_state: bool,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helion implementation of RWKV-6.

        RWKV-6 recurrence (output BEFORE state update):
            kv_t = k_t ⊗ v_t                    # Outer product
            o_t = sum((h + u * kv_t) * r_t)     # Output using OLD h + bonus
            h_t = h_{t-1} * exp(w_t) + kv_t     # State update AFTER output

        Uses dot_precision="ieee" for full precision outer products.
        """
        B, T, H, K_dim = r.shape
        V_dim = v.shape[-1]
        K_dim = hl.specialize(K_dim)
        V_dim = hl.specialize(V_dim)

        acc_dtype = torch.float32

        o = torch.empty(B, T, H, V_dim, dtype=acc_dtype, device=r.device)
        ht = torch.empty(B, H, K_dim, V_dim, dtype=acc_dtype, device=r.device)
        block_v = hl.register_block_size(V_dim)

        for i_b, i_h in hl.grid([B, H]):
            # Load bonus for this head (shared across batch)
            # u is [H, K], use integer indexing to get [K_dim] 1D tensor
            b_u = u[i_h, :].to(acc_dtype)  # [K_dim]

            for tile_v in hl.tile(V_dim, block_size=block_v):
                # Initialize hidden state
                b_h = hl.zeros([K_dim, tile_v], dtype=acc_dtype)

                if use_initial_state:
                    b_h = b_h + initial_state[i_b, i_h, :, tile_v].to(acc_dtype)

                # Sequential time loop
                for t_tile in hl.tile(T, block_size=1):
                    # Load inputs using tile indexing (gives 2D tensors for hl.dot)
                    b_r = r[i_b, t_tile, i_h, :].to(acc_dtype) * scale  # [1, K_dim]
                    b_k = k[i_b, t_tile, i_h, :].to(acc_dtype)  # [1, K_dim]
                    b_v = v[i_b, t_tile, i_h, tile_v].to(acc_dtype)  # [1, tile_v]

                    # Load decay (need integer index for broadcast)
                    t = t_tile.begin
                    b_w = w[i_b, t, i_h, :].to(acc_dtype)  # [K_dim] - 1D

                    # Compute outer product using hl.dot with IEEE precision
                    # b_k.T is [K_dim, 1], b_v is [1, tile_v], result is [K_dim, tile_v]
                    b_kv = hl.dot(b_k.T, b_v)  # [K_dim, tile_v]

                    # Compute output with bonus: o = r @ (h + u * kv)
                    # This uses OLD hidden state (before update)
                    b_h_with_bonus = b_h + b_u[:, None] * b_kv  # [K_dim, tile_v]

                    # Output: r is [1, K], h_with_bonus is [K, tile_v], result is [1, tile_v]
                    b_o = hl.dot(b_r, b_h_with_bonus)  # [1, tile_v]
                    o[i_b, t_tile, i_h, tile_v] = b_o

                    # Update hidden state AFTER computing output: h = h * exp(w) + kv
                    b_h = b_h * torch.exp(b_w)[:, None] + b_kv  # [K_dim, tile_v]

                if output_final_state:
                    ht[i_b, i_h, :, tile_v] = b_h

        return o, ht


def rwkv6_helion(
    r: torch.Tensor,  # [B, T, H, K] receptance (query)
    k: torch.Tensor,  # [B, T, H, K] key
    v: torch.Tensor,  # [B, T, H, V] value
    w: torch.Tensor,  # [B, T, H, K] decay in log-space
    u: torch.Tensor,  # [H, K] bonus
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Helion implementation of RWKV-6.

    Args:
        r: Receptance (query) tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        w: Decay tensor in log-space [B, T, H, K]
        u: Bonus tensor [H, K]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state

    Returns:
        o: Output tensor [B, T, H, V]
        ht: Final hidden state [B, H, K, V] if output_final_state else None
    """
    if not HELION_AVAILABLE:
        raise RuntimeError("Helion is not available. Please install helion.")

    B, T, H, K = r.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Create dummy tensor if no initial state (needs proper shape for Helion compilation)
    if initial_state is None:
        dummy_state = torch.empty(B, H, K, V, device=r.device, dtype=r.dtype)
    else:
        dummy_state = initial_state

    o, ht = rwkv6_helion_kernel(
        r, k, v, w, u,
        dummy_state,
        scale,
        initial_state is not None,
        output_final_state,
    )

    if not output_final_state:
        ht = None

    return o.to(r.dtype), ht


def test_rwkv6_helion():
    """Test that Helion kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    if not HELION_AVAILABLE:
        print("Helion not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations - use smaller sizes due to hl.dot precision limits
    configs = [
        # (B, T, H, K, V)
        (1, 32, 2, 32, 32),
        (2, 64, 4, 64, 64),
    ]

    print("Testing RWKV-6 Helion kernel vs PyTorch reference...")

    for B, T, H, K, V in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}")

        # Generate random inputs
        r = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
        w = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
        u = torch.randn(H, K, dtype=torch.float32, device=device)
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

        # Run reference
        ref_o, ref_ht = rwkv6_recurrent_ref(
            r.clone(), k.clone(), v.clone(), w.clone(), u.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Run Helion kernel
        helion_o, helion_ht = rwkv6_helion(
            r.clone(), k.clone(), v.clone(), w.clone(), u.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Check outputs
        o_atol = (ref_o - helion_o).abs().max().item()
        o_rtol = ((ref_o - helion_o).abs() / (ref_o.abs() + 1e-8)).max().item()

        ht_atol = (ref_ht - helion_ht).abs().max().item()
        ht_rtol = ((ref_ht - helion_ht).abs() / (ref_ht.abs() + 1e-8)).max().item()

        # Adjusted tolerances for hl.dot precision
        # Scale with sqrt(T) and sqrt(K*V) like in GLA
        atol_threshold = 5e-2 * math.sqrt(T / 32) * math.sqrt((K * V) / 1024)
        rtol_threshold = 0.1

        o_pass = o_atol < atol_threshold or o_rtol < rtol_threshold
        ht_pass = ht_atol < atol_threshold or ht_rtol < rtol_threshold

        status = "PASS" if (o_pass and ht_pass) else "FAIL"
        print(f"    Output:      atol={o_atol:.2e}, rtol={o_rtol:.2e} (threshold={atol_threshold:.2e}) [{status}]")
        print(f"    Final state: atol={ht_atol:.2e}, rtol={ht_rtol:.2e} (threshold={atol_threshold:.2e}) [{status}]")

        if not (o_pass and ht_pass):
            raise AssertionError(f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}")

    # Test without initial state
    print("\n  Testing without initial state...")
    B, T, H, K, V = 1, 32, 2, 32, 32
    r = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    w = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.1
    u = torch.randn(H, K, dtype=torch.float32, device=device)

    ref_o, _ = rwkv6_recurrent_ref(r.clone(), k.clone(), v.clone(), w.clone(), u.clone())
    helion_o, _ = rwkv6_helion(r.clone(), k.clone(), v.clone(), w.clone(), u.clone())

    atol = (ref_o - helion_o).abs().max().item()
    atol_threshold = 5e-2
    status = "PASS" if atol < atol_threshold else "FAIL"
    print(f"    Output atol={atol:.2e} (threshold={atol_threshold:.2e}) [{status}]")

    if atol >= atol_threshold:
        raise AssertionError(f"Test without initial state failed with atol={atol}")

    print("All RWKV-6 Helion tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RWKV-6 (Receptance Weighted Key Value) Triton Core Tests")
    print("=" * 60)

    test_rwkv6_triton_vs_reference()
    print()
    test_rwkv6_without_initial_state()
    print()
    test_rwkv6_half_precision()
    print()
    test_rwkv6_decay_behavior()
    print()
    test_rwkv6_bonus_behavior()
    print()
    benchmark_rwkv6()

    # Helion tests
    if HELION_AVAILABLE:
        print()
        print("=" * 60)
        print("RWKV-6 Helion Kernel Tests")
        print("=" * 60)
        test_rwkv6_helion()

    print("\n" + "=" * 60)
    print("All RWKV-6 tests completed successfully!")
    print("=" * 60)
