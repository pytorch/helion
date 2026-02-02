"""
Multi-Head Latent Attention (MLA) - Core Triton Kernel

==============================================================================
MATHEMATICAL CORE
==============================================================================

MLA compresses the KV cache to a low-dimensional latent space while using
decoupled RoPE (Rotary Position Embeddings) to preserve positional information.

Core Algorithm (Decoupled Attention):

    # KV Cache stores compressed latent + RoPE component
    c_kv:  [seq_len, d_c]    # Compressed KV latent (e.g., d_c=512)
    k_pe:  [seq_len, d_r]    # RoPE-encoded position keys (e.g., d_r=64)

    # Query has two components
    q_nope: [num_heads, d_c]  # Content query (no position encoding)
    q_pe:   [num_heads, d_r]  # Position query (RoPE-encoded)

    # Attention computation with decoupled scores
    score_content = Q_nope @ K_c^T           # Content-based attention
    score_position = Q_pe @ K_pe^T           # Position-based attention
    score = (score_content + score_position) / sqrt(d_c + d_r)

    # In MLA, the value is the same as the content key (compressed latent)
    V = K_c = c_kv

    # Output
    O = softmax(score) @ V

Key Insights:
    - KV cache compression: Store d_c + d_r per token instead of 2 * d_h * H
    - DeepSeek-V2/V3: d_c=512, d_r=64, d_h*H=16384 -> 32x compression
    - Decoupled RoPE allows position-agnostic content compression
    - Value reuses the compressed latent (no separate V storage)

Memory Layout (per token per layer):
    - Standard MHA:  2 * d_h * H elements (K and V)
    - MLA:          d_c + d_r elements (compressed KV + RoPE keys)

Complexity:
    - Time: O(seq_len^2 * (d_c + d_r) * H) for attention computation
    - KV Cache: O(seq_len * (d_c + d_r)) instead of O(seq_len * 2 * d_h * H)

References:
    - DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
      Language Model (arXiv:2405.04434)
    - FlashMLA: https://github.com/deepseek-ai/FlashMLA

==============================================================================
"""

import math

import torch
import triton
import triton.language as tl

import helion
import helion.language as hl


# ==============================================================================
# Triton Kernel: MLA Attention Forward (Dense, Non-Paged)
# ==============================================================================

@triton.jit
def mla_attention_fwd_kernel(
    # Pointers to tensors
    Q_nope,   # Query content: [B, H, d_c]
    Q_pe,     # Query position: [B, H, d_r]
    KV_c,     # Compressed KV (serves as both K content and V): [B, S, d_c]
    K_pe,     # Position keys: [B, S, d_r]
    O,        # Output: [B, H, d_c]
    LSE,      # Log-sum-exp for numerical stability: [B, H]
    # Scaling
    sm_scale,
    # Dimensions
    seq_len,
    num_heads,
    # Strides for Q_nope
    stride_qn_b, stride_qn_h, stride_qn_d,
    # Strides for Q_pe
    stride_qp_b, stride_qp_h, stride_qp_d,
    # Strides for KV_c
    stride_kvc_b, stride_kvc_s, stride_kvc_d,
    # Strides for K_pe
    stride_kpe_b, stride_kpe_s, stride_kpe_d,
    # Strides for O
    stride_o_b, stride_o_h, stride_o_d,
    # Stride for LSE
    stride_lse_b, stride_lse_h,
    # Block sizes
    BLOCK_H: tl.constexpr,    # Heads per block
    BLOCK_N: tl.constexpr,    # KV sequence positions per block
    D_C: tl.constexpr,        # Content dimension (d_c)
    D_R: tl.constexpr,        # RoPE dimension (d_r)
    RETURN_LSE: tl.constexpr, # Whether to return log-sum-exp
):
    """
    MLA Attention forward kernel with decoupled content and position attention.

    Grid: (batch, cdiv(num_heads, BLOCK_H))

    This kernel computes attention for a block of heads, iterating over
    all KV positions. The key insight is that attention scores come from
    two components:
        score = Q_nope @ KV_c^T + Q_pe @ K_pe^T

    And the value is the same as the content key (KV_c).

    Uses online softmax (Flash Attention style) for numerical stability.
    """
    # Get block indices
    i_batch = tl.program_id(0)
    i_head_block = tl.program_id(1)

    # Head indices for this block
    offs_h = i_head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads

    # Dimension offsets
    offs_d_c = tl.arange(0, D_C)
    offs_d_r = tl.arange(0, D_R)

    # Load Q_nope: [BLOCK_H, D_C]
    qn_ptrs = (Q_nope
               + i_batch * stride_qn_b
               + offs_h[:, None] * stride_qn_h
               + offs_d_c[None, :] * stride_qn_d)
    q_nope = tl.load(qn_ptrs, mask=mask_h[:, None], other=0.0)

    # Load Q_pe: [BLOCK_H, D_R]
    qp_ptrs = (Q_pe
               + i_batch * stride_qp_b
               + offs_h[:, None] * stride_qp_h
               + offs_d_r[None, :] * stride_qp_d)
    q_pe = tl.load(qp_ptrs, mask=mask_h[:, None], other=0.0)

    # Initialize online softmax accumulators
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")  # Max scores
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)                  # Sum of exp
    acc = tl.zeros([BLOCK_H, D_C], dtype=tl.float32)            # Output accumulator

    # Base pointers for KV
    kvc_base = KV_c + i_batch * stride_kvc_b
    kpe_base = K_pe + i_batch * stride_kpe_b

    # Iterate over KV sequence positions
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # Load KV_c block: [BLOCK_N, D_C] -> transpose for dot product
        kvc_ptrs = (kvc_base
                    + offs_n[:, None] * stride_kvc_s
                    + offs_d_c[None, :] * stride_kvc_d)
        kv_c = tl.load(kvc_ptrs, mask=mask_n[:, None], other=0.0)

        # Load K_pe block: [BLOCK_N, D_R]
        kpe_ptrs = (kpe_base
                    + offs_n[:, None] * stride_kpe_s
                    + offs_d_r[None, :] * stride_kpe_d)
        k_pe = tl.load(kpe_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute decoupled attention scores
        # score_content: [BLOCK_H, BLOCK_N] = Q_nope @ KV_c^T
        score_content = tl.dot(q_nope, tl.trans(kv_c))

        # score_position: [BLOCK_H, BLOCK_N] = Q_pe @ K_pe^T
        score_position = tl.dot(q_pe, tl.trans(k_pe))

        # Combined score with scaling
        qk = (score_content + score_position) * sm_scale

        # Mask invalid positions
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Scale factor for previous accumulator
        alpha = tl.exp(m_i - m_new)

        # Attention weights for current block
        p = tl.exp(qk - m_new[:, None])

        # Update sum of exponentials
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # Update output accumulator
        # V = KV_c (the compressed latent serves as value)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_c.dtype), kv_c)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output: [BLOCK_H, D_C]
    o_ptrs = (O
              + i_batch * stride_o_b
              + offs_h[:, None] * stride_o_h
              + offs_d_c[None, :] * stride_o_d)
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), mask=mask_h[:, None])

    # Optionally store log-sum-exp
    if RETURN_LSE:
        lse = m_i + tl.log(l_i)
        lse_ptrs = LSE + i_batch * stride_lse_b + offs_h * stride_lse_h
        tl.store(lse_ptrs, lse, mask=mask_h)


def mla_attention_fwd(
    q_nope: torch.Tensor,  # [B, H, d_c]
    q_pe: torch.Tensor,    # [B, H, d_r]
    kv_c: torch.Tensor,    # [B, S, d_c]
    k_pe: torch.Tensor,    # [B, S, d_r]
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Triton implementation of MLA Attention forward pass.

    This implements the core MLA attention with decoupled content and position
    components:
        score = Q_nope @ KV_c^T + Q_pe @ K_pe^T
        O = softmax(score / sqrt(d_c + d_r)) @ KV_c

    Note: In MLA, the value V is the same as the content key KV_c.

    Args:
        q_nope: Query content tensor [batch, num_heads, d_c]
        q_pe: Query position tensor [batch, num_heads, d_r]
        kv_c: Compressed KV cache (content keys and values) [batch, seq_len, d_c]
        k_pe: Position keys [batch, seq_len, d_r]
        sm_scale: Softmax scale (default: 1/sqrt(d_c + d_r))
        return_lse: Whether to return log-sum-exp values

    Returns:
        o: Output tensor [batch, num_heads, d_c]
        lse: Log-sum-exp tensor [batch, num_heads] if return_lse else None
    """
    B, H, d_c = q_nope.shape
    d_r = q_pe.shape[-1]
    S = kv_c.shape[1]

    # Validate shapes
    assert q_pe.shape == (B, H, d_r), f"q_pe shape mismatch: {q_pe.shape} vs {(B, H, d_r)}"
    assert kv_c.shape == (B, S, d_c), f"kv_c shape mismatch: {kv_c.shape} vs {(B, S, d_c)}"
    assert k_pe.shape == (B, S, d_r), f"k_pe shape mismatch: {k_pe.shape} vs {(B, S, d_r)}"

    if sm_scale is None:
        sm_scale = (d_c + d_r) ** -0.5

    # Allocate output
    o = torch.empty(B, H, d_c, dtype=q_nope.dtype, device=q_nope.device)
    lse = torch.empty(B, H, dtype=torch.float32, device=q_nope.device) if return_lse else None

    # Pad dimensions to power of 2 for Triton
    # Also ensure minimum of 16 for Triton's dot product (K dimension requirement)
    D_C = max(triton.next_power_of_2(d_c), 16)
    D_R = max(triton.next_power_of_2(d_r), 16)

    # Ensure dimensions are reasonable
    D_C = min(D_C, 512)
    D_R = min(D_R, 128)

    # Block sizes - reduce for larger dimensions to fit in shared memory
    # Shared memory usage ~ BLOCK_H * (D_C + D_R) + BLOCK_N * (D_C + D_R)
    if D_C >= 512:
        BLOCK_H = min(8, H)   # Smaller head block for large d_c
        BLOCK_N = 32          # Smaller KV block
    elif D_C >= 256:
        BLOCK_H = min(16, H)
        BLOCK_N = 32
    else:
        BLOCK_H = min(16, H)  # Heads per block
        BLOCK_N = 64          # KV positions per iteration

    # Grid dimensions
    grid = (B, triton.cdiv(H, BLOCK_H))

    # Launch kernel
    mla_attention_fwd_kernel[grid](
        q_nope, q_pe, kv_c, k_pe, o,
        lse if return_lse else o,  # Dummy pointer if not returning LSE
        sm_scale,
        S, H,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        kv_c.stride(0), kv_c.stride(1), kv_c.stride(2),
        k_pe.stride(0), k_pe.stride(1), k_pe.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0) if return_lse else 0,
        lse.stride(1) if return_lse else 0,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        D_C=D_C,
        D_R=D_R,
        RETURN_LSE=return_lse,
    )

    return o, lse


# ==============================================================================
# Helion Implementation
# ==============================================================================

@helion.kernel(static_shapes=True, autotune_effort="none")
def mla_attention_helion_kernel(
    q_nope: torch.Tensor,  # [B, H, d_c]
    q_pe: torch.Tensor,    # [B, H, d_r]
    kv_c: torch.Tensor,    # [B, S, d_c]
    k_pe: torch.Tensor,    # [B, S, d_r]
    lse_out: torch.Tensor, # [B, H] (may be unused if return_lse=False)
    sm_scale: float,
    return_lse: bool,
) -> torch.Tensor:
    """
    Helion implementation of MLA Attention forward pass.

    Implements the decoupled attention computation:
        score = Q_nope @ KV_c^T + Q_pe @ K_pe^T
        O = softmax(score * sm_scale) @ KV_c

    Note: Uses -1e10 instead of -inf for masked positions to avoid NaN when
    computing exp2(-inf - (-inf)) in the online softmax.

    Args:
        q_nope: Query content tensor [batch, num_heads, d_c]
        q_pe: Query position tensor [batch, num_heads, d_r]
        kv_c: Compressed KV cache (content keys and values) [batch, seq_len, d_c]
        k_pe: Position keys [batch, seq_len, d_r]
        lse_out: LSE output tensor [batch, num_heads]
        sm_scale: Softmax scale
        return_lse: Whether to write log-sum-exp values

    Returns:
        o: Output tensor [batch, num_heads, d_c]
    """
    B, H, d_c = q_nope.shape
    d_r = q_pe.shape[-1]
    S = kv_c.shape[1]

    # Specialize dimensions
    d_c = hl.specialize(d_c)
    d_r = hl.specialize(d_r)

    # Allocate output
    o = torch.empty(B, H, d_c, dtype=q_nope.dtype, device=q_nope.device)

    # Scale for exp2 instead of exp (1/log(2) = 1.44269504)
    qk_scale = sm_scale * 1.44269504

    # Use large negative value instead of -inf to avoid NaN in online softmax
    NEG_INF = -1e10

    # Process each batch and head
    for tile_b, tile_h in hl.tile([B, H], block_size=[1, None]):
        # Initialize online softmax accumulators
        m_i = hl.full([tile_h], NEG_INF, dtype=torch.float32)
        l_i = hl.zeros([tile_h], dtype=torch.float32)
        acc = hl.zeros([tile_h, d_c], dtype=torch.float32)

        # Load query content: [tile_h, d_c]
        q_nope_tile = q_nope[tile_b.begin, tile_h, :]

        # Load query position: [tile_h, d_r]
        q_pe_tile = q_pe[tile_b.begin, tile_h, :]

        # Iterate over KV sequence positions
        for tile_n in hl.tile(S):
            # Load KV_c block: [tile_n, d_c]
            kv_c_tile = kv_c[tile_b.begin, tile_n, :]

            # Load K_pe block: [tile_n, d_r]
            k_pe_tile = k_pe[tile_b.begin, tile_n, :]

            # Compute decoupled attention scores
            # Content score: [tile_h, tile_n] = q_nope @ kv_c^T
            qk_content = hl.zeros([tile_h, tile_n], dtype=torch.float32)
            qk_content = hl.dot(q_nope_tile, kv_c_tile.T, acc=qk_content)

            # Position score: [tile_h, tile_n] = q_pe @ k_pe^T
            qk_position = hl.zeros([tile_h, tile_n], dtype=torch.float32)
            qk_position = hl.dot(q_pe_tile, k_pe_tile.T, acc=qk_position)

            # Combined score
            qk = qk_content + qk_position

            # Mask invalid positions (positions beyond sequence length)
            offs_n = tile_n.begin + hl.arange(tile_n.block_size)
            valid_mask = offs_n[None, :] < S
            qk = torch.where(valid_mask, qk, NEG_INF)

            # Online softmax update
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            # V = KV_c (the compressed latent serves as value)
            p = p.to(kv_c_tile.dtype)
            acc = hl.dot(p, kv_c_tile, acc=acc)

            m_i = m_ij

        # Normalize output - handle case where l_i is very small
        l_i = torch.maximum(l_i, torch.full_like(l_i, 1e-10))
        acc = acc / l_i[:, None]

        # Store output: [tile_h, d_c]
        o[tile_b, tile_h, :] = acc[None, :, :].to(o.dtype)

        # Store LSE if requested
        if return_lse:
            # LSE = m_i / qk_scale + log2(l_i) / log2(e) = m_i / qk_scale + log(l_i)
            lse_val = m_i / qk_scale + torch.log(l_i)
            lse_out[tile_b, tile_h] = lse_val[None, :]

    return o


def mla_attention_helion(
    q_nope: torch.Tensor,  # [B, H, d_c]
    q_pe: torch.Tensor,    # [B, H, d_r]
    kv_c: torch.Tensor,    # [B, S, d_c]
    k_pe: torch.Tensor,    # [B, S, d_r]
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Helion-based implementation of MLA Attention forward pass.

    Args:
        q_nope: Query content tensor [batch, num_heads, d_c]
        q_pe: Query position tensor [batch, num_heads, d_r]
        kv_c: Compressed KV cache (content keys and values) [batch, seq_len, d_c]
        k_pe: Position keys [batch, seq_len, d_r]
        sm_scale: Softmax scale (default: 1/sqrt(d_c + d_r))
        return_lse: Whether to return log-sum-exp values

    Returns:
        o: Output tensor [batch, num_heads, d_c]
        lse: Log-sum-exp tensor [batch, num_heads] if return_lse else None
    """
    B, H, d_c = q_nope.shape
    d_r = q_pe.shape[-1]

    if sm_scale is None:
        sm_scale = (d_c + d_r) ** -0.5

    # Allocate LSE tensor (always allocated, but only used if return_lse=True)
    lse_out = torch.empty(B, H, dtype=torch.float32, device=q_nope.device)

    o = mla_attention_helion_kernel(q_nope, q_pe, kv_c, k_pe, lse_out, sm_scale, return_lse)

    return o, lse_out if return_lse else None


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def mla_attention_ref(
    q_nope: torch.Tensor,  # [B, H, d_c]
    q_pe: torch.Tensor,    # [B, H, d_r]
    kv_c: torch.Tensor,    # [B, S, d_c]
    k_pe: torch.Tensor,    # [B, S, d_r]
    sm_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch reference implementation of MLA Attention.

    Implements the decoupled attention computation:
        score = Q_nope @ KV_c^T + Q_pe @ K_pe^T
        O = softmax(score / sqrt(d_c + d_r)) @ KV_c

    Args:
        q_nope: Query content tensor [batch, num_heads, d_c]
        q_pe: Query position tensor [batch, num_heads, d_r]
        kv_c: Compressed KV cache [batch, seq_len, d_c]
        k_pe: Position keys [batch, seq_len, d_r]
        sm_scale: Softmax scale (default: 1/sqrt(d_c + d_r))
        return_lse: Whether to return log-sum-exp values

    Returns:
        o: Output tensor [batch, num_heads, d_c]
        lse: Log-sum-exp tensor [batch, num_heads] if return_lse else None
    """
    B, H, d_c = q_nope.shape
    d_r = q_pe.shape[-1]

    if sm_scale is None:
        sm_scale = (d_c + d_r) ** -0.5

    # Compute decoupled attention scores
    # Content score: [B, H, S] = [B, H, d_c] @ [B, d_c, S]
    score_content = torch.bmm(q_nope, kv_c.transpose(-2, -1))

    # Position score: [B, H, S] = [B, H, d_r] @ [B, d_r, S]
    score_position = torch.bmm(q_pe, k_pe.transpose(-2, -1))

    # Combined score with scaling
    scores = (score_content + score_position) * sm_scale

    # Compute log-sum-exp if needed
    lse = None
    if return_lse:
        lse = torch.logsumexp(scores, dim=-1)

    # Softmax
    attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)

    # Apply attention to values (V = KV_c)
    # Output: [B, H, d_c] = [B, H, S] @ [B, S, d_c]
    output = torch.bmm(attn_weights, kv_c)

    return output, lse


# ==============================================================================
# Numerical Tests
# ==============================================================================

def test_mla_triton_vs_reference():
    """Test that Triton kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, H, S, d_c, d_r)
    # Typical MLA configs from DeepSeek-V2/V3
    # Note: Triton dot requires K >= 16, so d_c and d_r must be >= 16
    configs = [
        # Basic tests
        (1, 16, 64, 64, 16),     # Small config
        (2, 32, 128, 128, 32),   # Medium config
        # DeepSeek-like configurations
        (1, 128, 256, 512, 64),  # DeepSeek-V2 style
        (2, 128, 512, 512, 64),  # Longer sequence
        (4, 64, 128, 256, 32),   # Multiple batches
        # Smaller but valid configs (d_c, d_r >= 16 for Triton dot)
        (1, 8, 32, 32, 16),      # Small dimensions
        (1, 1, 16, 64, 16),      # Single head
        (2, 16, 64, 64, 32),     # Different aspect ratio
    ]

    print("Testing MLA Triton kernel vs PyTorch reference...")

    for B, H, S, d_c, d_r in configs:
        compression = (2 * 128 * H) / (d_c + d_r)  # Assuming d_h=128
        print(f"  Config: B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r} (~{compression:.1f}x compression)")

        # Generate random inputs
        q_nope = torch.randn(B, H, d_c, dtype=torch.float32, device=device)
        q_pe = torch.randn(B, H, d_r, dtype=torch.float32, device=device)
        kv_c = torch.randn(B, S, d_c, dtype=torch.float32, device=device)
        k_pe = torch.randn(B, S, d_r, dtype=torch.float32, device=device)

        # Run reference
        ref_o, ref_lse = mla_attention_ref(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Run Triton kernel
        tri_o, tri_lse = mla_attention_fwd(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Check output
        atol_o = (ref_o - tri_o).abs().max().item()
        rtol_o = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

        # Check LSE
        atol_lse = (ref_lse - tri_lse).abs().max().item()

        # Tolerances for online softmax
        atol_threshold = 1e-2
        passed = atol_o < atol_threshold
        status = "PASS" if passed else "FAIL"

        print(f"    Output: atol={atol_o:.2e}, rtol={rtol_o:.2e}, LSE atol={atol_lse:.2e} [{status}]")

        if not passed:
            raise AssertionError(
                f"Test failed for config B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}"
            )

    print("All output tests passed!")


def test_mla_half_precision():
    """Test MLA with half precision inputs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # DeepSeek-style config
    B, H, S, d_c, d_r = 2, 128, 256, 512, 64

    print(f"Testing MLA half precision: B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}")

    # Generate in half precision
    q_nope = torch.randn(B, H, d_c, dtype=torch.float16, device=device)
    q_pe = torch.randn(B, H, d_r, dtype=torch.float16, device=device)
    kv_c = torch.randn(B, S, d_c, dtype=torch.float16, device=device)
    k_pe = torch.randn(B, S, d_r, dtype=torch.float16, device=device)

    # Run reference (convert to float32 for accuracy)
    ref_o, _ = mla_attention_ref(
        q_nope.float(), q_pe.float(),
        kv_c.float(), k_pe.float(),
    )
    ref_o = ref_o.to(torch.float16)

    # Run Triton kernel
    tri_o, _ = mla_attention_fwd(
        q_nope, q_pe, kv_c, k_pe,
    )

    atol = (ref_o - tri_o).abs().max().item()
    print(f"  Half precision: atol={atol:.2e}")

    # Larger tolerance for half precision
    assert atol < 0.1, f"Test failed with atol={atol}"
    print("  PASS")


def test_mla_bfloat16():
    """Test MLA with bfloat16 inputs (commonly used in LLMs)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # DeepSeek-style config
    B, H, S, d_c, d_r = 2, 128, 256, 512, 64

    print(f"Testing MLA bfloat16: B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}")

    # Generate in bfloat16
    q_nope = torch.randn(B, H, d_c, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_r, dtype=torch.bfloat16, device=device)
    kv_c = torch.randn(B, S, d_c, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(B, S, d_r, dtype=torch.bfloat16, device=device)

    # Run reference (convert to float32 for accuracy)
    ref_o, _ = mla_attention_ref(
        q_nope.float(), q_pe.float(),
        kv_c.float(), k_pe.float(),
    )
    ref_o = ref_o.to(torch.bfloat16)

    # Run Triton kernel
    tri_o, _ = mla_attention_fwd(
        q_nope, q_pe, kv_c, k_pe,
    )

    atol = (ref_o - tri_o).abs().max().item()
    print(f"  bfloat16: atol={atol:.2e}")

    # Larger tolerance for bfloat16
    assert atol < 0.1, f"Test failed with atol={atol}"
    print("  PASS")


def test_mla_kv_cache_compression():
    """Demonstrate the KV cache compression achieved by MLA."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    print("\nMLA KV Cache Compression Analysis:")
    print("-" * 60)

    # DeepSeek-V2/V3 configuration
    configs = [
        ("Standard MHA (baseline)", 128, 128, None, None),
        ("DeepSeek-V2/V3 MLA", 128, 128, 512, 64),
        ("Hypothetical MLA (smaller)", 128, 64, 256, 32),
    ]

    seq_len = 4096
    num_layers = 60  # DeepSeek-V2 has 60 layers
    dtype_size = 2   # bfloat16 = 2 bytes

    for name, num_heads, head_dim, d_c, d_r in configs:
        if d_c is None:  # MHA baseline
            kv_cache_per_token = 2 * num_heads * head_dim * dtype_size  # K and V
            cache_description = f"2 * H * d = 2 * {num_heads} * {head_dim}"
        else:  # MLA
            kv_cache_per_token = (d_c + d_r) * dtype_size  # Compressed KV + RoPE
            cache_description = f"d_c + d_r = {d_c} + {d_r}"

        total_cache = kv_cache_per_token * seq_len * num_layers

        print(f"  {name}:")
        print(f"    KV cache per token: {cache_description} = {kv_cache_per_token} bytes")
        print(f"    Total cache ({seq_len} tokens, {num_layers} layers): {total_cache / 1024 / 1024:.2f} MB")

        if d_c is not None:
            mha_cache = 2 * num_heads * head_dim * dtype_size * seq_len * num_layers
            compression = mha_cache / total_cache
            print(f"    Compression vs MHA: {compression:.1f}x")
        print()


def test_mla_decoupled_attention():
    """Verify the decoupled attention mechanism works correctly."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Use d_c >= 16 and d_r >= 16 for Triton dot product compatibility
    B, H, S, d_c, d_r = 1, 4, 16, 32, 16

    print("Testing MLA decoupled attention mechanism...")

    # Generate inputs
    q_nope = torch.randn(B, H, d_c, dtype=torch.float32, device=device)
    q_pe = torch.randn(B, H, d_r, dtype=torch.float32, device=device)
    kv_c = torch.randn(B, S, d_c, dtype=torch.float32, device=device)
    k_pe = torch.randn(B, S, d_r, dtype=torch.float32, device=device)

    sm_scale = (d_c + d_r) ** -0.5

    # Manual computation to verify decoupling
    # Content score
    score_content = torch.bmm(q_nope, kv_c.transpose(-2, -1))
    # Position score
    score_position = torch.bmm(q_pe, k_pe.transpose(-2, -1))
    # Combined
    manual_scores = (score_content + score_position) * sm_scale
    manual_attn = torch.softmax(manual_scores, dim=-1)
    manual_output = torch.bmm(manual_attn, kv_c)

    # Using reference function
    ref_output, _ = mla_attention_ref(q_nope, q_pe, kv_c, k_pe)

    # Using Triton kernel
    tri_output, _ = mla_attention_fwd(q_nope, q_pe, kv_c, k_pe)

    # Verify all match
    manual_vs_ref = (manual_output - ref_output).abs().max().item()
    ref_vs_tri = (ref_output - tri_output).abs().max().item()

    print(f"  Manual vs Reference: atol={manual_vs_ref:.2e}")
    print(f"  Reference vs Triton: atol={ref_vs_tri:.2e}")

    assert manual_vs_ref < 1e-5, "Manual computation doesn't match reference"
    assert ref_vs_tri < 1e-2, "Triton doesn't match reference"

    print("  Decoupled attention verified correctly!")


def benchmark_mla():
    """Simple benchmark comparing Triton vs PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    torch.manual_seed(42)
    device = torch.device("cuda")

    # DeepSeek-V2 style config
    B, H, S, d_c, d_r = 4, 128, 2048, 512, 64

    print(f"\nBenchmark (B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}):")

    q_nope = torch.randn(B, H, d_c, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, H, d_r, dtype=torch.bfloat16, device=device)
    kv_c = torch.randn(B, S, d_c, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(B, S, d_r, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        mla_attention_fwd(q_nope, q_pe, kv_c, k_pe)
    torch.cuda.synchronize()

    # Benchmark Triton
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        mla_attention_fwd(q_nope, q_pe, kv_c, k_pe)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000

    # Benchmark reference
    start = time.time()
    for _ in range(n_iters):
        mla_attention_ref(q_nope, q_pe, kv_c, k_pe)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iters * 1000

    # Calculate FLOPS
    flops = B * H * S * (d_c + d_r) * 2  # QK^T
    flops += B * H * S * d_c  # softmax @ V
    flops *= 2  # multiply-add = 2 ops

    # Calculate memory bandwidth
    bytes_accessed = (
        B * H * (d_c + d_r) +  # Q
        B * S * (d_c + d_r) +  # KV
        B * H * d_c            # O
    ) * 2  # bfloat16

    print(f"  Triton:    {triton_time:.2f} ms ({flops / 1e9 / triton_time:.1f} TFLOPS, {bytes_accessed / 1e6 / triton_time:.1f} GB/s)")
    print(f"  Reference: {ref_time:.2f} ms ({flops / 1e9 / ref_time:.1f} TFLOPS, {bytes_accessed / 1e6 / ref_time:.1f} GB/s)")
    print(f"  Speedup:   {ref_time / triton_time:.2f}x")


# ==============================================================================
# Helion Tests
# ==============================================================================

def test_mla_helion_vs_reference():
    """Test that Helion kernel matches PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, H, S, d_c, d_r)
    configs = [
        # Basic tests
        (1, 16, 64, 64, 16),     # Small config
        (2, 32, 128, 128, 32),   # Medium config
        # DeepSeek-like configurations
        (1, 128, 256, 512, 64),  # DeepSeek-V2 style
        (2, 128, 512, 512, 64),  # Longer sequence
        (4, 64, 128, 256, 32),   # Multiple batches
        # Smaller but valid configs
        (1, 8, 32, 32, 16),      # Small dimensions
        (1, 1, 16, 64, 16),      # Single head
        (2, 16, 64, 64, 32),     # Different aspect ratio
    ]

    print("Testing MLA Helion kernel vs PyTorch reference...")

    for B, H, S, d_c, d_r in configs:
        print(f"  Config: B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}")

        # Generate random inputs
        q_nope = torch.randn(B, H, d_c, dtype=torch.float32, device=device)
        q_pe = torch.randn(B, H, d_r, dtype=torch.float32, device=device)
        kv_c = torch.randn(B, S, d_c, dtype=torch.float32, device=device)
        k_pe = torch.randn(B, S, d_r, dtype=torch.float32, device=device)

        # Run reference
        ref_o, ref_lse = mla_attention_ref(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Run Helion kernel
        helion_o, helion_lse = mla_attention_helion(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Check output
        atol_o = (ref_o - helion_o).abs().max().item()

        # Tolerances for online softmax
        atol_threshold = 1e-2
        passed = atol_o < atol_threshold
        status = "PASS" if passed else "FAIL"

        print(f"    Output: atol={atol_o:.2e} [{status}]")

        if not passed:
            raise AssertionError(
                f"Test failed for config B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}"
            )

    print("All Helion vs Reference tests passed!")


def test_mla_helion_vs_triton():
    """Test that Helion kernel matches Triton kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Test configurations: (B, H, S, d_c, d_r)
    configs = [
        (1, 16, 64, 64, 16),
        (2, 32, 128, 128, 32),
        (1, 128, 256, 512, 64),
        (2, 16, 64, 64, 32),
    ]

    print("Testing MLA Helion kernel vs Triton kernel...")

    for B, H, S, d_c, d_r in configs:
        print(f"  Config: B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}")

        # Generate random inputs
        q_nope = torch.randn(B, H, d_c, dtype=torch.float32, device=device)
        q_pe = torch.randn(B, H, d_r, dtype=torch.float32, device=device)
        kv_c = torch.randn(B, S, d_c, dtype=torch.float32, device=device)
        k_pe = torch.randn(B, S, d_r, dtype=torch.float32, device=device)

        # Run Triton kernel
        tri_o, tri_lse = mla_attention_fwd(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Run Helion kernel
        helion_o, helion_lse = mla_attention_helion(
            q_nope.clone(), q_pe.clone(),
            kv_c.clone(), k_pe.clone(),
            return_lse=True,
        )

        # Check output
        atol_o = (tri_o - helion_o).abs().max().item()

        # Tolerances for online softmax
        atol_threshold = 1e-2
        passed = atol_o < atol_threshold
        status = "PASS" if passed else "FAIL"

        print(f"    Output: atol={atol_o:.2e} [{status}]")

        if not passed:
            raise AssertionError(
                f"Test failed for config B={B}, H={H}, S={S}, d_c={d_c}, d_r={d_r}"
            )

    print("All Helion vs Triton tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("MLA (Multi-Head Latent Attention) Triton Core Tests")
    print("=" * 60)

    test_mla_triton_vs_reference()
    print()
    test_mla_half_precision()
    print()
    test_mla_bfloat16()
    print()
    test_mla_decoupled_attention()
    print()
    test_mla_kv_cache_compression()
    print()
    test_mla_helion_vs_reference()
    print()
    test_mla_helion_vs_triton()
    print()
    benchmark_mla()

    print("\n" + "=" * 60)
    print("All MLA tests completed successfully!")
    print("=" * 60)
