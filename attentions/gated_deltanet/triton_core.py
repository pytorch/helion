"""
Gated DeltaNet (Gated Delta Rule) - Core Triton Kernel

==============================================================================
MATHEMATICAL CORE
==============================================================================

Gated DeltaNet implements error-correcting memory updates with decay gates.
Unlike standard linear attention that can only add associations, the delta rule
enables memory **modification** by correcting retrieval errors.

Recurrence Form (per head):
    h_t = exp(g_t) * h_{t-1} + k_t ⊗ (β_t * (v_t - h_{t-1}^T @ k_t))
    o_t = scale * h_t @ q_t

Where:
    - h_t ∈ R^{K×V} is the hidden state (key-value memory matrix)
    - g_t ∈ R (scalar) is the decay gate in log-space
    - k_t ∈ R^K is the key vector
    - v_t ∈ R^V is the value vector
    - q_t ∈ R^K is the query vector
    - β_t ∈ R (scalar) is the update strength gate
    - scale = 1/√K is the normalization factor
    - o_t ∈ R^V is the output

Key components:
    - exp(g_t): Decay gate for selective forgetting
    - v_t - h_{t-1}^T @ k_t: The DELTA/ERROR term - difference between target
      value v_t and what the memory would predict (h @ k)
    - β_t: Update strength gate controlling how much to correct
    - k_t ⊗ (...): Outer product update to memory

The delta rule allows memory MODIFICATION (not just addition) by:
    1. Computing what memory would retrieve: predicted = h @ k
    2. Computing error: delta = v - predicted
    3. Applying gated correction: h += k ⊗ (β * delta)

This error-correcting mechanism improves over standard linear attention which
suffers from "memory overload" - accumulating retrieval errors as the model
can only add, never modify, associations.

Complexity:
    - Time: O(T × K × V) per head (linear in sequence length)
    - Space: O(K × V) for hidden state (constant in sequence length)

References:
    - Gated Delta Networks: Improving Mamba2 with Delta Rule
      (Yang et al., ICLR 2025)
    - https://arxiv.org/abs/2412.06464
    - https://github.com/NVlabs/GatedDeltaNet

==============================================================================
"""

import torch
import triton
import triton.language as tl

import helion
import helion.language as hl


# ==============================================================================
# Triton Kernel: Fused Recurrent Gated Delta Rule
# ==============================================================================

@triton.jit
def gated_deltanet_recurrent_fwd_kernel(
    # Pointers to matrices
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,      # Decay gate in log-space [B, T, H]
    beta_ptr,   # Update strength gate [B, T, H]
    o_ptr,
    h0_ptr,     # Initial state (optional)
    ht_ptr,     # Final state output (optional)
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
    Fused recurrent Gated DeltaNet forward pass.

    Computes for each timestep t:
        # Apply decay
        h_t = exp(g_t) * h_{t-1}
        # Compute delta/error
        predicted = h_t @ k_t  (sum over K)
        delta = v_t - predicted
        # Update with gated correction
        h_t = h_t + k_t ⊗ (β_t * delta)
        # Compute output
        o_t = scale * (h_t @ q_t)

    Grid: (num_v_blocks, batch * heads)
    """
    # Get block indices
    i_v = tl.program_id(0)   # Which V block
    i_bh = tl.program_id(1)  # Which (batch, head) combination

    i_b = i_bh // H
    i_h = i_bh % H

    # Compute offsets for K and V dimensions
    o_k = tl.arange(0, BK)                  # [BK]
    o_v = i_v * BV + tl.arange(0, BV)       # [BV]

    # Masks for boundary conditions
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]       # [BK, BV]

    # Initialize hidden state block
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # Load initial state if provided
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    # Starting positions for this batch/head
    # q, k: [B, T, H, K] layout
    # v: [B, T, H, V] layout
    # g, beta: [B, T, H] layout
    base_qk = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    base_g = i_b * T * H + i_h

    # Iterate through timesteps
    for t in range(T):
        # Load inputs for timestep t
        # q: [B, T, H, K] -> load [BK]
        p_q = q_ptr + base_qk + t * H * K + o_k
        b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32) * scale

        # k: [B, T, H, K] -> load [BK]
        p_k = k_ptr + base_qk + t * H * K + o_k
        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)

        # v: [B, T, H, V] -> load [BV]
        p_v = v_ptr + base_v + t * H * V + o_v
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)

        # g: [B, T, H] -> load scalar
        p_g = g_ptr + base_g + t * H
        b_g = tl.load(p_g).to(tl.float32)

        # beta: [B, T, H] -> load scalar
        p_beta = beta_ptr + base_g + t * H
        b_beta = tl.load(p_beta).to(tl.float32)

        # Apply decay gate: h = h * exp(g)
        b_h = b_h * tl.exp(b_g)  # [BK, BV]

        # Compute delta/error: predicted value vs actual value
        # predicted = h @ k (for each v dimension, sum over k)
        # h: [BK, BV], k: [BK] -> sum over BK to get [BV]
        b_predicted = tl.sum(b_h * b_k[:, None], axis=0)  # [BV]
        b_delta = b_v - b_predicted  # [BV]

        # Apply gated correction: h = h + k ⊗ (β * delta)
        b_v_corrected = b_beta * b_delta  # [BV]
        b_h = b_h + b_k[:, None] * b_v_corrected[None, :]  # [BK, BV]

        # Compute output: o = h @ q (sum over K dimension)
        # h: [BK, BV], q: [BK] -> sum over BK to get [BV]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)  # [BV]

        # Store output
        p_o = o_ptr + (i_b * T * H * V) + (t * H * V) + (i_h * V) + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

    # Store final state if requested
    if STORE_FINAL_STATE:
        p_ht = ht_ptr + i_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


def gated_deltanet_recurrent_fwd(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    g: torch.Tensor,     # [B, T, H] decay gate in log-space
    beta: torch.Tensor,  # [B, T, H] update strength gate
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Triton implementation of Gated DeltaNet forward pass.

    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Decay gate in log-space [B, T, H]
        beta: Update strength gate [B, T, H]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state

    Returns:
        o: Output tensor [B, T, H, V]
        ht: Final hidden state [B, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Block sizes - K must fit in single block for correctness
    # (we need to compute full h @ k sum)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)

    # Output buffer
    o = q.new_empty(B, T, H, V, dtype=torch.float32)

    # Final state buffer
    ht = q.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None

    # Launch kernel
    grid = (NV, B * H)
    gated_deltanet_recurrent_fwd_kernel[grid](
        q, k, v, g, beta, o,
        initial_state, ht,
        B, T, H, K, V,
        scale,
        BK, BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
    )

    return o.to(q.dtype), ht


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def gated_deltanet_recurrent_ref(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    g: torch.Tensor,     # [B, T, H] decay gate in log-space
    beta: torch.Tensor,  # [B, T, H] update strength gate
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch reference implementation of Gated DeltaNet.

    This is a straightforward loop-based implementation that matches
    the mathematical definition exactly.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    if scale is None:
        scale = K ** -0.5

    # Convert to float32 for numerical stability
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    # Initialize hidden state: h ∈ R^{B×H×K×V}
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state.float().clone()

    # Output buffer
    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    # Recurrence loop
    for t in range(T):
        # Extract timestep t inputs
        q_t = q[:, t] * scale  # [B, H, K]
        k_t = k[:, t]          # [B, H, K]
        v_t = v[:, t]          # [B, H, V]
        g_t = g[:, t]          # [B, H]
        beta_t = beta[:, t]    # [B, H]

        # Apply decay gate: h = h * exp(g)
        h = h * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, H, K, V]

        # Compute predicted value: h @ k (sum over K dimension)
        # h: [B, H, K, V], k: [B, H, K] -> [B, H, V]
        predicted = (h * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, V]

        # Compute delta/error
        delta = v_t - predicted  # [B, H, V]

        # Apply gated correction: v_corrected = beta * delta
        v_corrected = beta_t.unsqueeze(-1) * delta  # [B, H, V]

        # Update memory: h = h + k ⊗ v_corrected
        kv = k_t.unsqueeze(-1) * v_corrected.unsqueeze(-2)  # [B, H, K, V]
        h = h + kv

        # Compute output: o = h @ q (contract over K dimension)
        o[:, t] = (q_t.unsqueeze(-1) * h).sum(dim=-2)  # [B, H, V]

    if not output_final_state:
        h = None

    return o.to(dtype), h


# ==============================================================================
# Helion Kernel Implementation
# ==============================================================================

@helion.kernel(static_shapes=True, autotune_effort="none")
def gated_deltanet_helion_kernel(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    g: torch.Tensor,     # [B, T, H] decay gate in log-space
    beta: torch.Tensor,  # [B, T, H] update strength gate
    initial_state: torch.Tensor,  # [B, H, K, V] or dummy tensor if not used
    scale: float,
    use_initial_state: bool,
    output_final_state: bool,
) -> torch.Tensor:
    """
    Helion implementation of Gated DeltaNet forward pass.

    Matches the Triton kernel interface exactly with the same
    mathematical operations. Follows the blocking strategy from
    the gdn_fwd_h example: Grid over (batch, heads), tile over V,
    iterate over T with chunk_size=1.

    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Decay gate in log-space [B, T, H]
        beta: Update strength gate [B, T, H]
        initial_state: Initial hidden state [B, H, K, V] (or dummy if not used)
        scale: Scaling factor (1/sqrt(K))
        use_initial_state: Whether to use the initial state
        output_final_state: Whether to return final hidden state

    Returns:
        Output tensor [B, T, H, V] and final state [B, H, K, V]
    """
    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    K_dim = hl.specialize(K_dim)
    V_dim = hl.specialize(V_dim)

    acc_dtype = torch.float32
    dtype = q.dtype

    # Allocate outputs inside kernel
    o = torch.empty(B, T, H, V_dim, dtype=acc_dtype, device=q.device)
    ht = torch.empty(B, H, K_dim, V_dim, dtype=acc_dtype, device=q.device)

    # Register block size for V dimension
    block_v = hl.register_block_size(V_dim)

    # Grid over batch and heads (following gdn example pattern)
    for i_b, i_h in hl.grid([B, H]):
        # Tile over V dimension
        for tile_v in hl.tile(V_dim, block_size=block_v):
            # Initialize hidden state [K, tile_v]
            # Always use hl.zeros first (following gdn_fwd_h pattern) then add initial state
            b_h = hl.zeros([K_dim, tile_v], dtype=acc_dtype)
            if use_initial_state:
                b_h += initial_state[i_b, i_h, :, tile_v].to(acc_dtype)

            # Iterate through timesteps (block_size=1 for sequential processing)
            for t_tile in hl.tile(T, block_size=1):
                # Get the integer index for this timestep (like gdn example's t_i_last)
                t = t_tile.begin

                # Load inputs for this timestep using integer indexing for scalars
                # and tile indexing for vectors
                b_q = q[i_b, t_tile, i_h, :].to(acc_dtype) * scale  # [1, K]
                b_k = k[i_b, t_tile, i_h, :].to(acc_dtype)  # [1, K]
                b_v = v[i_b, t_tile, i_h, tile_v].to(acc_dtype)  # [1, tile_v]
                # Use integer index for scalars (like gdn example)
                b_g = g[i_b, t, i_h].to(acc_dtype)  # scalar
                b_beta = beta[i_b, t, i_h].to(acc_dtype)  # scalar

                # Apply decay gate: h = h * exp(g)
                # Following gdn_fwd_h pattern: compute exp first, then use in-place multiply
                b_g_exp = torch.exp(b_g)  # scalar
                b_h *= b_g_exp  # [K, tile_v] *= scalar

                # Compute delta/error using hl.dot pattern
                # predicted = h @ k (sum over K): [K, tile_v] @ [K, 1] -> [tile_v, 1] -> [1, tile_v]
                # b_k is [1, K], need to transpose for the sum
                c_h = b_h.to(dtype)
                b_predicted = hl.dot(b_k, c_h, out_dtype=acc_dtype)  # [1, K] @ [K, tile_v] = [1, tile_v]
                b_delta = b_v - b_predicted  # [1, tile_v]

                # Apply gated correction: h = h + k ⊗ (beta * delta)
                # b_beta is scalar, broadcasts to [1, tile_v]
                b_v_corrected = b_beta * b_delta  # [1, tile_v]
                b_v_corrected = b_v_corrected.to(dtype)
                b_h = hl.dot(b_k.T, b_v_corrected, acc=b_h)  # [K, 1] @ [1, tile_v] + [K, tile_v] = [K, tile_v]

                # Compute output: o = h @ q (sum over K)
                c_h = b_h.to(dtype)
                b_o = hl.dot(b_q, c_h, out_dtype=acc_dtype)  # [1, K] @ [K, tile_v] = [1, tile_v]

                # Store output
                o[i_b, t_tile, i_h, tile_v] = b_o

            # Store final state
            if output_final_state:
                ht[i_b, i_h, :, tile_v] = b_h

    return o, ht


def gated_deltanet_helion(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    g: torch.Tensor,     # [B, T, H] decay gate in log-space
    beta: torch.Tensor,  # [B, T, H] update strength gate
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Helion-based implementation of Gated DeltaNet forward pass.

    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Decay gate in log-space [B, T, H]
        beta: Update strength gate [B, T, H]
        scale: Scaling factor (default: 1/sqrt(K))
        initial_state: Initial hidden state [B, H, K, V]
        output_final_state: Whether to return final hidden state

    Returns:
        o: Output tensor [B, T, H, V]
        ht: Final hidden state [B, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Create dummy initial state if not provided
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(B, H, K, V, dtype=q.dtype, device=q.device)

    # Run Helion kernel
    o, ht = gated_deltanet_helion_kernel(
        q, k, v, g, beta, initial_state, scale, use_initial_state, output_final_state
    )

    # Convert output dtype and handle final state
    if not output_final_state:
        ht = None

    return o.to(q.dtype), ht


# ==============================================================================
# Numerical Tests
# ==============================================================================

def test_gated_deltanet_triton_vs_reference():
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

    print("Testing Gated DeltaNet Triton kernel vs PyTorch reference...")

    for B, T, H, K, V in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}")

        # Generate random inputs
        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        # Normalize k for stability (common practice)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
        # Decay gate in log-space (negative for decay)
        g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
        # Update strength gate (typically in [0, 1])
        beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.1

        # Run reference
        ref_o, ref_ht = gated_deltanet_recurrent_ref(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Run Triton kernel
        tri_o, tri_ht = gated_deltanet_recurrent_fwd(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Check outputs
        o_atol = (ref_o - tri_o).abs().max().item()
        o_rtol = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

        ht_atol = (ref_ht - tri_ht).abs().max().item()
        ht_rtol = ((ref_ht - tri_ht).abs() / (ref_ht.abs() + 1e-8)).max().item()

        # Tolerances
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


def test_gated_deltanet_without_initial_state():
    """Test Gated DeltaNet without initial state."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 64, 4, 64, 64

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))

    # Run without initial state
    ref_o, _ = gated_deltanet_recurrent_ref(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone()
    )
    tri_o, _ = gated_deltanet_recurrent_fwd(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone()
    )

    atol = (ref_o - tri_o).abs().max().item()
    print(f"Test without initial state: atol={atol:.2e}")
    assert atol < 1e-4, f"Test failed with atol={atol}"
    print("  PASS")


def test_gated_deltanet_half_precision():
    """Test Gated DeltaNet with half precision inputs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 64, 4, 64, 64

    # Generate in float16
    q = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float16, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device)
    g = (torch.randn(B, T, H, device=device) * 0.1).to(torch.float16)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device)).to(torch.float16)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.1

    # Run reference (in float32 internally)
    ref_o, ref_ht = gated_deltanet_recurrent_ref(
        q.float(), k.float(), v.float(), g.float(), beta.float(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run Triton kernel
    tri_o, tri_ht = gated_deltanet_recurrent_fwd(
        q, k, v, g, beta,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    atol = (ref_o.to(tri_o.dtype) - tri_o).abs().max().item()
    print(f"Test half precision: atol={atol:.2e}")
    # Larger tolerance for half precision
    assert atol < 5e-2, f"Test failed with atol={atol}"
    print("  PASS")


def test_gated_deltanet_delta_rule_behavior():
    """Test that delta rule actually corrects memory errors."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 1, 10, 1, 16, 16

    # Create a simple scenario:
    # - First, store a k-v association
    # - Then, try to overwrite with a new value for the same key
    # - The delta rule should correct the memory

    q = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.zeros(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.zeros(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.zeros(B, T, H, dtype=torch.float32, device=device)  # No decay
    beta = torch.ones(B, T, H, dtype=torch.float32, device=device)  # Full update

    # Set up a key
    key_pattern = torch.randn(K, device=device)
    key_pattern = key_pattern / key_pattern.norm()

    # First timestep: store association k -> v1
    k[:, 0, 0, :] = key_pattern
    v1 = torch.randn(V, device=device)
    v[:, 0, 0, :] = v1

    # Second timestep: try to overwrite k -> v2
    k[:, 1, 0, :] = key_pattern
    v2 = torch.randn(V, device=device)
    v[:, 1, 0, :] = v2

    # Query at timestep 2
    q[:, 2, 0, :] = key_pattern

    _, ht = gated_deltanet_recurrent_fwd(
        q, k, v, g, beta,
        output_final_state=True,
    )

    # After delta rule, querying with key should retrieve v2, not v1
    # because the delta rule corrects the memory
    retrieved = (ht[0, 0] * key_pattern.unsqueeze(-1)).sum(dim=0)  # [V]

    # Check that retrieved is closer to v2 than v1
    dist_v1 = (retrieved - v1).norm().item()
    dist_v2 = (retrieved - v2).norm().item()

    print(f"Test delta rule behavior:")
    print(f"  Distance to v1 (old): {dist_v1:.4f}")
    print(f"  Distance to v2 (new): {dist_v2:.4f}")
    print(f"  Delta rule working: {dist_v2 < dist_v1}")

    assert dist_v2 < dist_v1, "Delta rule should make retrieved value closer to v2"
    print("  PASS")


def benchmark_gated_deltanet():
    """Simple benchmark comparing Triton vs PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 4, 1024, 8, 128, 128

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))

    # Warmup
    for _ in range(3):
        gated_deltanet_recurrent_fwd(q, k, v, g, beta)
    torch.cuda.synchronize()

    # Benchmark Triton
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        gated_deltanet_recurrent_fwd(q, k, v, g, beta)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000

    # Benchmark reference
    start = time.time()
    for _ in range(n_iters):
        gated_deltanet_recurrent_ref(q, k, v, g, beta)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iters * 1000

    print(f"\nBenchmark (B={B}, T={T}, H={H}, K={K}, V={V}):")
    print(f"  Triton:    {triton_time:.2f} ms")
    print(f"  Reference: {ref_time:.2f} ms")
    print(f"  Speedup:   {ref_time / triton_time:.2f}x")


# ==============================================================================
# Helion Tests
# ==============================================================================

def test_gated_deltanet_helion_vs_reference():
    """Test that Helion kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, T, H, K, V)
    configs = [
        (1, 32, 2, 32, 32),
        (2, 64, 4, 64, 64),
        (1, 128, 4, 64, 128),
    ]

    print("Testing Gated DeltaNet Helion kernel vs PyTorch reference...")

    for B, T, H, K, V in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}")

        # Generate random inputs
        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.nn.functional.normalize(
            torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
        )
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
        g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
        beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.1

        # Run reference
        ref_o, ref_ht = gated_deltanet_recurrent_ref(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        # Run Helion kernel
        try:
            helion_o, helion_ht = gated_deltanet_helion(
                q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
                initial_state=h0.clone(),
                output_final_state=True,
            )

            # Check outputs
            o_atol = (ref_o - helion_o).abs().max().item()
            ht_atol = (ref_ht - helion_ht).abs().max().item()

            # Tolerances - slightly relaxed for accumulated numerical error with longer sequences
            atol_threshold = 2e-2

            o_pass = o_atol < atol_threshold
            ht_pass = ht_atol < atol_threshold

            status = "PASS" if (o_pass and ht_pass) else "FAIL"
            print(f"    Output: atol={o_atol:.2e}, Final state: atol={ht_atol:.2e} [{status}]")

            if not (o_pass and ht_pass):
                raise AssertionError(
                    f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}"
                )
        except Exception as e:
            print(f"    SKIPPED ({type(e).__name__}: {e})")
            raise

    print("All Helion vs reference tests passed!")


def test_gated_deltanet_helion_vs_triton():
    """Test that Helion kernel matches Triton kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 64, 4, 64, 64

    print(f"Testing Gated DeltaNet Helion vs Triton: B={B}, T={T}, H={H}, K={K}, V={V}")

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
    )
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device) * 0.1

    # Run Triton kernel
    triton_o, triton_ht = gated_deltanet_recurrent_fwd(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Run Helion kernel
    try:
        helion_o, helion_ht = gated_deltanet_helion(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            initial_state=h0.clone(),
            output_final_state=True,
        )

        o_atol = (triton_o - helion_o).abs().max().item()
        ht_atol = (triton_ht - helion_ht).abs().max().item()

        print(f"  Output: Helion vs Triton atol={o_atol:.2e}")
        print(f"  Final state: Helion vs Triton atol={ht_atol:.2e}")

        assert o_atol < 1e-2, f"Helion vs Triton output mismatch: atol={o_atol}"
        assert ht_atol < 1e-2, f"Helion vs Triton state mismatch: atol={ht_atol}"
        print("  PASS")
    except Exception as e:
        print(f"  SKIPPED ({type(e).__name__}: {e})")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Gated DeltaNet (Gated Delta Rule) Triton Core Tests")
    print("=" * 60)

    test_gated_deltanet_triton_vs_reference()
    print()
    test_gated_deltanet_without_initial_state()
    print()
    test_gated_deltanet_half_precision()
    print()
    test_gated_deltanet_delta_rule_behavior()
    print()

    print("=" * 60)
    print("Gated DeltaNet Helion Tests")
    print("=" * 60)
    print()
    test_gated_deltanet_helion_vs_reference()
    print()
    test_gated_deltanet_helion_vs_triton()
    print()

    benchmark_gated_deltanet()

    print("\n" + "=" * 60)
    print("All Gated DeltaNet tests completed successfully!")
    print("=" * 60)
