"""
Sliding Window Attention (SWA) - Core Triton Kernel

==============================================================================
MATHEMATICAL CORE
==============================================================================

Sliding Window Attention restricts each token to attend only to a local window
of neighboring tokens, creating a banded attention pattern.

Core Algorithm:
    M[i,j] = 0       if |i - j| <= W/2
           = -inf    otherwise

    O = softmax((Q @ K^T / sqrt(d)) + M) @ V

Where:
    - W = window size (total window width)
    - W/2 = half_window (tokens attended on each side)
    - Token at position i attends to positions in [i - W/2, i + W/2]
    - The attention matrix becomes a banded matrix with bandwidth W

Complexity:
    - Standard attention: O(N^2 * d) time, O(N^2) memory
    - Sliding window:     O(N * W * d) time, O(N * W) memory

Trade-off:
    Information from distant tokens propagates through L/W layers.
    Mitigated by attention sinks (Mistral), global tokens (Longformer),
    or interleaved full/SWA layers (SWAA).

References:
    - Longformer: The Long-Document Transformer (Beltagy et al., 2020)
    - FlashAttention (Dao et al., 2022)
    - Mistral 7B (Jiang et al., 2023) - SWA + attention sinks

==============================================================================
"""

import math

import torch
import triton
import triton.language as tl

import helion
import helion.language as hl


# ==============================================================================
# Triton Kernel: Sliding Window Attention Forward
# ==============================================================================

@triton.jit
def swa_attention_fwd_kernel(
    # Pointers to tensors
    Q,  # Query tensor
    K,  # Key tensor
    V,  # Value tensor
    O,  # Output tensor
    # Softmax scale
    sm_scale,
    # Dimensions
    seq_len,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Window parameters
    half_window: tl.constexpr,  # W/2: how many tokens to attend on each side
    # Block sizes
    BLOCK_M: tl.constexpr,  # Block size for query sequence
    BLOCK_N: tl.constexpr,  # Block size for key/value sequence
    BLOCK_D: tl.constexpr,  # Block size for head dimension
    # Attention mode
    IS_CAUSAL: tl.constexpr,  # Whether to also apply causal masking
):
    """
    Sliding Window Attention forward kernel.

    Grid: (batch, num_heads, cdiv(seq_len, BLOCK_M))

    For each query position i, attends only to key positions j where:
        |i - j| <= half_window

    When IS_CAUSAL is True, additionally requires j <= i (causal mask).

    Uses online softmax (Flash Attention style) for numerical stability
    and memory efficiency.
    """
    # Get block indices
    i_batch = tl.program_id(0)  # Batch index
    i_head = tl.program_id(1)   # Head index
    i_m = tl.program_id(2)      # Query block index

    # Compute query position offsets
    offs_m = i_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Query positions
    offs_n = tl.arange(0, BLOCK_N)                   # KV positions (iterated)
    offs_d = tl.arange(0, BLOCK_D)                   # Head dimension

    # Mask for valid query positions
    mask_m = offs_m < seq_len

    # Base pointers with batch and head offsets
    q_base = Q + i_batch * stride_qb + i_head * stride_qh
    k_base = K + i_batch * stride_kb + i_head * stride_kh
    v_base = V + i_batch * stride_vb + i_head * stride_vh
    o_base = O + i_batch * stride_ob + i_head * stride_oh

    # Load query block: [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max scores
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # Sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)         # Output accumulator

    # Compute the window bounds for this query block
    # Query positions are: i_m * BLOCK_M to (i_m + 1) * BLOCK_M - 1
    # For the first query in the block, the left bound is: (i_m * BLOCK_M) - half_window
    # For the last query in the block, the right bound is: ((i_m + 1) * BLOCK_M - 1) + half_window

    # Start of KV iteration: max(0, first_query_pos - half_window)
    start_n = tl.maximum(0, i_m * BLOCK_M - half_window)
    # Align to BLOCK_N for efficiency
    start_n = (start_n // BLOCK_N) * BLOCK_N

    # End of KV iteration
    end_n = seq_len
    if IS_CAUSAL:
        # Causal: can't look beyond query position
        end_n = tl.minimum(end_n, (i_m + 1) * BLOCK_M)
    else:
        # Non-causal: last_query_pos + half_window
        end_n = tl.minimum(end_n, (i_m + 1) * BLOCK_M + half_window)

    # Iterate over key/value blocks within the window
    for block_start_n in range(start_n, end_n, BLOCK_N):
        # Current KV positions
        curr_n = block_start_n + offs_n

        # Mask for valid KV positions
        mask_n = curr_n < seq_len

        # Build the sliding window mask: |query_pos - key_pos| <= half_window
        # For each query position offs_m[i], valid key positions are:
        #   offs_m[i] - half_window <= curr_n[j] <= offs_m[i] + half_window
        window_mask = (curr_n[None, :] >= (offs_m[:, None] - half_window)) & \
                      (curr_n[None, :] <= (offs_m[:, None] + half_window))

        # Combine with causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_n[None, :]
            combined_mask = window_mask & causal_mask & mask_n[None, :]
        else:
            combined_mask = window_mask & mask_n[None, :]

        # Load key block: [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + curr_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute attention scores: QK^T / sqrt(d)
        # q: [BLOCK_M, BLOCK_D], k: [BLOCK_N, BLOCK_D]
        # qk: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply combined mask (set invalid positions to -inf)
        qk = tl.where(combined_mask, qk, float("-inf"))

        # Online softmax update
        # New max per query position
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Compute scaling factor for previous accumulator
        # Handle -inf case: when m_i is -inf, alpha should be 0 (no previous contribution)
        # When m_new is -inf (all scores are -inf), use 1.0 to avoid NaN
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))

        # Compute attention weights for current block
        # When m_new is -inf, qk - m_new would be NaN, so we handle it
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, tl.exp(qk - m_new[:, None]))

        # Update sum of exponentials
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # Load value block: [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + curr_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Update output accumulator
        # Scale previous accumulator and add weighted values
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Normalize output
    # Handle case where l_i is zero (no valid attention positions)
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    # Store output: [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), mask=mask_m[:, None])


def swa_attention_fwd(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    window_size: int,  # Total window size W
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Triton implementation of Sliding Window Attention forward pass.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        window_size: Total window size W (token attends to W/2 on each side)
        is_causal: Whether to also apply causal masking (j <= i)
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        o: Output tensor [batch, num_heads, seq_len, head_dim]
    """
    B, H, S, D = q.shape

    if sm_scale is None:
        sm_scale = D ** -0.5

    half_window = window_size // 2

    # Allocate output
    o = torch.empty_like(q)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)

    # Ensure BLOCK_D is reasonable
    BLOCK_D = min(BLOCK_D, 128)

    # Grid dimensions
    grid = (B, H, triton.cdiv(S, BLOCK_M))

    # Launch kernel
    swa_attention_fwd_kernel[grid](
        q, k, v, o,
        sm_scale,
        S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        half_window=half_window,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
    )

    return o


# ==============================================================================
# Helion Implementation
# ==============================================================================

@helion.kernel(static_shapes=True, autotune_effort="none")
def swa_attention_helion_kernel(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    half_window: int,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """
    Helion implementation of Sliding Window Attention.

    Uses block_size=1 for batch/head dimensions (following flex_attention pattern)
    for proper mask broadcasting.

    Note: Uses -1e10 instead of -inf for masked positions to avoid NaN when
    computing exp2(-inf - (-inf)) in the online softmax. This happens when
    entire key blocks are masked out (e.g., query position 40 cannot attend
    to keys 0-31 with a window of ±8).

    Args:
        q: Query tensor [B, H, S, D]
        k: Key tensor [B, H, S, D]
        v: Value tensor [B, H, S, D]
        half_window: Half of window size (tokens attended on each side)
        is_causal: Whether to apply causal masking
        sm_scale: Softmax scale factor

    Returns:
        Output tensor [B, H, S, D]
    """
    B, H, S, D = q.shape
    D = hl.specialize(D)
    half_window = hl.specialize(half_window)

    # Output tensor
    o = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)

    # Scale for exp2 instead of exp (1/log(2) = 1.44269504)
    qk_scale = sm_scale * 1.44269504

    # Use large negative value instead of -inf to avoid NaN in online softmax
    # when entire key blocks are masked (exp2(-inf - (-inf)) = NaN)
    NEG_INF = -1e10

    # Process each batch, head, and query tile (flex_attention pattern)
    for tile_b, tile_h, tile_m in hl.tile([B, H, S], block_size=[1, 1, None]):
        # Initialize online softmax accumulators
        m_i = hl.full([tile_m], NEG_INF, dtype=torch.float32)
        l_i = hl.zeros([tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_m, D], dtype=torch.float32)

        # Load query tile: [tile_m, D]
        q_tile = q[tile_b.begin, tile_h.begin, tile_m, :]

        # Iterate over key/value positions
        for tile_n in hl.tile(S):
            # Load key tile: [tile_n, D] -> transpose to [D, tile_n]
            k_tile = k[tile_b.begin, tile_h.begin, tile_n, :]  # [tile_n, D]
            k_tile_t = k_tile.T  # [D, tile_n]

            # Compute attention scores using hl.dot: [tile_m, D] @ [D, tile_n] = [tile_m, tile_n]
            qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            qk = hl.dot(q_tile, k_tile_t, acc=qk)

            # Create position indices for masking
            offs_m = tile_m.begin + hl.arange(tile_m.block_size)
            offs_n = tile_n.begin + hl.arange(tile_n.block_size)

            # Sliding window mask: |query_pos - key_pos| <= half_window
            # Upper bound: key <= query + half_window
            upper_ok = offs_n[None, :] <= (offs_m[:, None] + half_window)
            # Lower bound: key >= query - half_window (reformulated to avoid subtraction issues)
            # key >= query - hw  <==>  key + hw >= query
            lower_ok = (offs_n[None, :] + half_window) >= offs_m[:, None]
            window_mask = upper_ok & lower_ok

            # Apply causal mask if needed: query >= key
            if is_causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                window_mask = window_mask & causal_mask

            qk = torch.where(window_mask, qk, NEG_INF)

            # Online softmax update
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            # Load values and accumulate
            v_tile = v[tile_b.begin, tile_h.begin, tile_n, :]  # [tile_n, D]
            p = p.to(v_tile.dtype)
            acc = hl.dot(p, v_tile, acc=acc)

            m_i = m_ij

        # Normalize output - handle case where l_i is very small
        l_i = torch.maximum(l_i, torch.full_like(l_i, 1e-10))
        acc = acc / l_i[:, None]
        o[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(o.dtype)

    return o


def swa_attention_helion(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    window_size: int,
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Helion-based implementation of Sliding Window Attention.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        window_size: Total window size W
        is_causal: Whether to apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    B, H, S, D = q.shape

    if sm_scale is None:
        sm_scale = D ** -0.5

    half_window = window_size // 2

    return swa_attention_helion_kernel(q, k, v, half_window, is_causal, sm_scale)


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def swa_attention_ref(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    window_size: int,  # Total window size W
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation of Sliding Window Attention.

    Creates a banded attention mask where position i can attend to
    positions j where |i - j| <= window_size // 2.
    """
    B, H, S, D = q.shape

    if sm_scale is None:
        sm_scale = D ** -0.5

    half_window = window_size // 2

    # Compute attention scores: [B, H, S, S]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Create sliding window mask
    # Position indices
    row_idx = torch.arange(S, device=q.device).view(S, 1)  # [S, 1]
    col_idx = torch.arange(S, device=q.device).view(1, S)  # [1, S]

    # Window mask: |i - j| <= half_window
    window_mask = (col_idx >= row_idx - half_window) & (col_idx <= row_idx + half_window)

    # Apply causal mask if needed
    if is_causal:
        causal_mask = col_idx <= row_idx
        combined_mask = window_mask & causal_mask
    else:
        combined_mask = window_mask

    # Apply mask
    attn_scores = attn_scores.masked_fill(~combined_mask, float("-inf"))

    # Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Handle NaN from all-inf rows (shouldn't happen with valid window)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Apply attention to values: [B, H, S, D]
    output = torch.matmul(attn_weights, v)

    return output


# ==============================================================================
# Numerical Tests
# ==============================================================================

def test_swa_triton_vs_reference():
    """Test that Triton kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, H, S, D, W)
    configs = [
        # Basic configurations
        (1, 4, 64, 64, 16),     # Small window
        (1, 4, 128, 64, 32),    # Medium window
        (2, 8, 256, 64, 64),    # Larger sequence
        (1, 4, 128, 64, 128),   # Window equals sequence length (full attention)
        # Different head dimensions
        (1, 4, 64, 32, 16),
        (1, 4, 64, 128, 16),
        # Edge cases
        (1, 1, 64, 64, 8),      # Single head
        (2, 4, 64, 64, 4),      # Very small window
    ]

    print("Testing SWA Triton kernel vs PyTorch reference...")

    for B, H, S, D, W in configs:
        half_w = W // 2
        print(f"  Config: B={B}, H={H}, S={S}, D={D}, W={W} (half={half_w})")

        # Generate random inputs
        q = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.float32, device=device)

        # Test both causal and non-causal
        for is_causal in [False, True]:
            causal_str = "causal" if is_causal else "non-causal"

            # Run reference
            ref_o = swa_attention_ref(
                q.clone(), k.clone(), v.clone(),
                window_size=W,
                is_causal=is_causal,
            )

            # Run Triton kernel
            tri_o = swa_attention_fwd(
                q.clone(), k.clone(), v.clone(),
                window_size=W,
                is_causal=is_causal,
            )

            # Check outputs
            atol = (ref_o - tri_o).abs().max().item()
            rtol = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

            # Tolerances - flash attention style algorithms have some numerical
            # differences due to online softmax
            atol_threshold = 5e-3
            passed = atol < atol_threshold
            status = "PASS" if passed else "FAIL"

            print(f"    {causal_str}: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

            if not passed:
                raise AssertionError(
                    f"Test failed for config B={B}, H={H}, S={S}, D={D}, "
                    f"W={W}, is_causal={is_causal}"
                )

    print("All tests passed!")


def test_swa_half_precision():
    """Test SWA with half precision inputs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, H, S, D, W = 2, 8, 128, 64, 32

    print(f"Testing SWA half precision: B={B}, H={H}, S={S}, D={D}, W={W}")

    # Generate in half precision
    q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)

    # Run reference (convert to float32 for accuracy)
    ref_o = swa_attention_ref(
        q.float(), k.float(), v.float(),
        window_size=W,
        is_causal=True,
    ).to(torch.float16)

    # Run Triton kernel
    tri_o = swa_attention_fwd(
        q, k, v,
        window_size=W,
        is_causal=True,
    )

    atol = (ref_o - tri_o).abs().max().item()
    print(f"  Half precision: atol={atol:.2e}")

    # Larger tolerance for half precision
    assert atol < 0.05, f"Test failed with atol={atol}"
    print("  PASS")


def test_swa_window_boundary():
    """Test SWA at window boundaries to ensure correct masking."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("Testing SWA window boundary behavior...")

    # Small example where we can inspect the attention pattern
    B, H, S, D = 1, 1, 8, 16
    W = 4  # half_window = 2

    # Create inputs with distinct patterns
    q = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float32, device=device)

    # Reference and Triton
    ref_o = swa_attention_ref(q, k, v, window_size=W, is_causal=False)
    tri_o = swa_attention_fwd(q, k, v, window_size=W, is_causal=False)

    atol = (ref_o - tri_o).abs().max().item()
    print(f"  Small example (S={S}, W={W}): atol={atol:.2e}")

    assert atol < 5e-3, f"Boundary test failed with atol={atol}"
    print("  PASS")


def test_swa_vs_full_attention():
    """Test that SWA with large window equals full attention."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, H, S, D = 2, 4, 64, 64

    print(f"Testing SWA with full window vs standard attention...")

    q = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float32, device=device)

    # Full attention (standard scaled dot-product)
    sm_scale = D ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    full_attn_o = torch.matmul(attn_weights, v)

    # SWA with window >= 2 * (S - 1) should equal full attention
    # Window of 2*S ensures all positions are within window
    swa_o = swa_attention_fwd(q, k, v, window_size=2 * S, is_causal=False)

    atol = (full_attn_o - swa_o).abs().max().item()
    print(f"  SWA (W={2*S}) vs full attention: atol={atol:.2e}")

    assert atol < 5e-3, f"Full window test failed with atol={atol}"
    print("  PASS")


def test_swa_memory_pattern():
    """Visualize and test the sliding window attention pattern."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    print("\nSliding Window Attention Pattern Analysis:")
    print("-" * 50)

    S = 8  # Small sequence for visualization
    W = 4  # Window size
    half_w = W // 2

    # Create attention mask pattern
    row_idx = torch.arange(S).view(S, 1)
    col_idx = torch.arange(S).view(1, S)

    window_mask = (col_idx >= row_idx - half_w) & (col_idx <= row_idx + half_w)
    causal_window_mask = window_mask & (col_idx <= row_idx)

    print(f"Sequence length: {S}, Window size: {W} (half: {half_w})")
    print("\nNon-causal sliding window mask:")
    print("  Position: ", end="")
    for j in range(S):
        print(f" {j}", end="")
    print()

    for i in range(S):
        print(f"  Q[{i}]:     ", end="")
        for j in range(S):
            print(" 1" if window_mask[i, j] else " 0", end="")
        print(f"  (attends to K[{max(0, i-half_w)}-{min(S-1, i+half_w)}])")

    print("\nCausal sliding window mask:")
    for i in range(S):
        print(f"  Q[{i}]:     ", end="")
        for j in range(S):
            print(" 1" if causal_window_mask[i, j] else " 0", end="")
        print(f"  (attends to K[{max(0, i-half_w)}-{i}])")

    # Compute complexity savings
    full_attn_ops = S * S
    swa_ops = sum(min(i + half_w + 1, S) - max(0, i - half_w) for i in range(S))
    causal_swa_ops = sum(i + 1 - max(0, i - half_w) for i in range(S))

    print(f"\nComplexity analysis (S={S}, W={W}):")
    print(f"  Full attention ops:       {full_attn_ops}")
    print(f"  SWA ops:                  {swa_ops} ({100*swa_ops/full_attn_ops:.1f}%)")
    print(f"  Causal SWA ops:           {causal_swa_ops} ({100*causal_swa_ops/full_attn_ops:.1f}%)")


def benchmark_swa():
    """Simple benchmark comparing Triton SWA vs full attention."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, H, S, D = 4, 8, 1024, 64
    W = 256  # Window size

    q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(3):
        swa_attention_fwd(q, k, v, window_size=W, is_causal=True)
    torch.cuda.synchronize()

    # Benchmark SWA
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        swa_attention_fwd(q, k, v, window_size=W, is_causal=True)
    torch.cuda.synchronize()
    swa_time = (time.time() - start) / n_iters * 1000

    # Benchmark reference (full materialization - slow)
    start = time.time()
    for _ in range(n_iters):
        swa_attention_ref(q, k, v, window_size=W, is_causal=True)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iters * 1000

    print(f"\nBenchmark (B={B}, H={H}, S={S}, D={D}, W={W}):")
    print(f"  Triton SWA:  {swa_time:.2f} ms")
    print(f"  Reference:   {ref_time:.2f} ms")
    print(f"  Speedup:     {ref_time / swa_time:.2f}x")


# ==============================================================================
# Helion Tests
# ==============================================================================

def test_swa_helion_vs_reference():
    """Test that Helion kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, H, S, D, W)
    configs = [
        (1, 4, 64, 64, 16),
        (1, 4, 128, 64, 32),
        (2, 4, 64, 64, 16),
    ]

    print("Testing SWA Helion kernel vs PyTorch reference...")

    for B, H, S, D, W in configs:
        half_w = W // 2
        print(f"  Config: B={B}, H={H}, S={S}, D={D}, W={W} (half={half_w})")

        q = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.float32, device=device)

        for is_causal in [False, True]:
            causal_str = "causal" if is_causal else "non-causal"

            ref_o = swa_attention_ref(
                q.clone(), k.clone(), v.clone(),
                window_size=W, is_causal=is_causal,
            )

            try:
                helion_o = swa_attention_helion(
                    q.clone(), k.clone(), v.clone(),
                    window_size=W, is_causal=is_causal,
                )

                atol = (ref_o - helion_o).abs().max().item()
                atol_threshold = 2e-2
                passed = atol < atol_threshold
                status = "PASS" if passed else "FAIL"

                print(f"    {causal_str}: atol={atol:.2e} [{status}]")

                if not passed:
                    raise AssertionError(
                        f"Test failed for config B={B}, H={H}, S={S}, D={D}, W={W}, is_causal={is_causal}"
                    )
            except Exception as e:
                print(f"    {causal_str}: SKIPPED ({type(e).__name__}: {e})")
                raise

    print("All Helion vs reference tests passed!")


def test_swa_helion_vs_triton():
    """Test that Helion kernel matches Triton kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    B, H, S, D, W = 2, 4, 64, 64, 16

    print(f"Testing SWA Helion vs Triton: B={B}, H={H}, S={S}, D={D}, W={W}")

    q = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float32, device=device)

    for is_causal in [False, True]:
        causal_str = "causal" if is_causal else "non-causal"

        tri_o = swa_attention_fwd(
            q.clone(), k.clone(), v.clone(),
            window_size=W, is_causal=is_causal,
        )

        helion_o = swa_attention_helion(
            q.clone(), k.clone(), v.clone(),
            window_size=W, is_causal=is_causal,
        )

        atol = (tri_o - helion_o).abs().max().item()
        print(f"  {causal_str}: Helion vs Triton atol={atol:.2e}")

        assert atol < 2e-2, f"Helion vs Triton mismatch: atol={atol}"

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("SWA (Sliding Window Attention) Triton Core Tests")
    print("=" * 60)

    test_swa_triton_vs_reference()
    print()
    test_swa_half_precision()
    print()
    test_swa_window_boundary()
    print()
    test_swa_vs_full_attention()
    print()
    test_swa_memory_pattern()
    print()
    benchmark_swa()

    print("\n" + "=" * 60)
    print("All SWA tests completed successfully!")
    print("=" * 60)
