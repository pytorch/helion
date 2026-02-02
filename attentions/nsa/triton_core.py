"""
Native Sparse Attention (NSA) - Core Triton Kernels

==============================================================================
MATHEMATICAL CORE
==============================================================================

NSA computes sparse attention through three complementary branches combined
with learned gates:

    o_t = g_cmp * Attn(q_t, K_cmp, V_cmp)   # Compressed (coarse global)
        + g_slc * Attn(q_t, K_slc, V_slc)   # Selected (fine-grained important)
        + g_swa * Attn(q_t, K_swa, V_swa)   # Sliding window (local)

Where:
    - g_cmp, g_slc, g_swa are learned gate scores in [0, 1]
    - Each branch uses standard scaled dot-product attention

Branch 1: Compressed Attention (Coarse-Grained Global)
------------------------------------------------------
Mean-pool K, V into blocks of size BS, creating compressed representations:

    K_cmp[i] = mean(K[i*BS:(i+1)*BS])  # [num_blocks, H, D]
    V_cmp[i] = mean(V[i*BS:(i+1)*BS])  # [num_blocks, H, D]

Attention over compressed KV (causal - only completed blocks):

    NC_t = floor((t+1) / BS)  # number of complete blocks before position t
    o_cmp_t = softmax(q_t @ K_cmp[:NC_t]^T / sqrt(d)) @ V_cmp[:NC_t]

Branch 2: Selected Attention (Fine-Grained Important)
----------------------------------------------------
Use compression attention scores to identify important blocks:

    importance = softmax(q_t @ K_cmp^T)  # attention weights as importance
    block_indices = top_k(importance, S)  # select top-S blocks

Then compute fine-grained attention over selected blocks:

    K_slc = gather(K, block_indices * BS, block_size=BS)
    V_slc = gather(V, block_indices * BS, block_size=BS)
    o_slc_t = softmax(q_t @ K_slc^T / sqrt(d)) @ V_slc

Branch 3: Sliding Window Attention (Local Context)
--------------------------------------------------
Standard causal sliding window attention:

    o_swa_t = softmax(q_t @ K[max(0, t-W+1):t+1]^T / sqrt(d)) @ V[max(0, t-W+1):t+1]

Complexity Analysis:
    - Compression: O(T * C * d) where C = T/BS << T
    - Selection: O(T * S * BS * d) where S << C
    - Sliding Window: O(T * W * d)
    - Total: O(T * (C + S*BS + W) * d) vs O(T^2 * d) for standard attention

For 64k context with S=16, BS=64, W=64:
    - NSA: ~130M operations
    - Standard: ~4B operations
    - ~30x theoretical speedup

References:
    - Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
      (Yuan et al., DeepSeek, 2025)
    - https://arxiv.org/abs/2502.11089

==============================================================================
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

# Helion imports
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False


# ==============================================================================
# Triton Kernel: Compression Attention (Mean-Pooled Blocks)
# ==============================================================================

@triton.jit
def nsa_compression_attn_kernel(
    # Input/Output pointers
    q_ptr,       # Query: [B, T, H, K]
    k_cmp_ptr,   # Compressed keys: [B, C, H, K] where C = ceil(T/BS)
    v_cmp_ptr,   # Compressed values: [B, C, H, V]
    o_ptr,       # Output: [B, T, H, V]
    lse_ptr,     # Log-sum-exp: [B, T, H] for top-k selection
    # Dimensions
    B,           # Batch size
    T,           # Sequence length
    C,           # Number of compressed blocks
    H: tl.constexpr,  # Number of heads
    K: tl.constexpr,  # Key dimension
    V: tl.constexpr,  # Value dimension
    BS: tl.constexpr, # Block size for compression
    # Scaling
    scale,
    # Block sizes for tiling
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr, # Block of compressed tokens to process
):
    """
    Compressed attention kernel - attention over mean-pooled KV blocks.

    For each query position t, attends to completed compression blocks:
        NC = (t + 1) // BS  (number of complete blocks)
        o_t = softmax(q_t @ K_cmp[:NC]^T / sqrt(d)) @ V_cmp[:NC]

    Grid: (T, B * H)
    """
    # Get indices
    i_t = tl.program_id(0)   # Token position
    i_bh = tl.program_id(1)  # Batch * head index

    i_b = i_bh // H
    i_h = i_bh % H

    # Number of complete compression blocks before position t (causal masking)
    NC = (i_t + 1) // BS

    # If no complete blocks yet, output zeros
    if NC == 0:
        # Store zeros
        for i_v in range(0, V, BV):
            o_v = i_v + tl.arange(0, BV)
            m_v = o_v < V
            p_o = o_ptr + i_b * T * H * V + i_t * H * V + i_h * V + o_v
            tl.store(p_o, tl.zeros([BV], dtype=tl.float32), mask=m_v)
        # Store -inf for LSE
        p_lse = lse_ptr + i_b * T * H + i_t * H + i_h
        tl.store(p_lse, float('-inf'))
        return

    # Load query vector [K]
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    p_q = q_ptr + i_b * T * H * K + i_t * H * K + i_h * K + o_k
    b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32) * scale

    # Online softmax variables
    m_max = float('-inf')  # Running max
    l_sum = 0.0            # Running sum of exp(x - max)
    b_o = tl.zeros([BV], dtype=tl.float32)  # Accumulated output

    # Iterate over compression blocks
    for i_c in range(0, NC, BC):
        n_c = min(BC, NC - i_c)  # Actual number of blocks in this iteration

        # Load compressed keys [BC, K] for blocks i_c to i_c + n_c
        for j in range(n_c):
            c_idx = i_c + j

            # Load K_cmp[c_idx]
            p_k = k_cmp_ptr + i_b * C * H * K + c_idx * H * K + i_h * K + o_k
            b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)

            # Compute attention score
            s = tl.sum(b_q * b_k, axis=0)  # dot product

            # Online softmax update
            m_new = tl.maximum(m_max, s)
            exp_diff = tl.exp(m_max - m_new)
            exp_s = tl.exp(s - m_new)

            # Update accumulators
            l_sum = l_sum * exp_diff + exp_s

            # Load V_cmp[c_idx] and update output
            o_v = tl.arange(0, BV)
            m_v = o_v < V
            p_v = v_cmp_ptr + i_b * C * H * V + c_idx * H * V + i_h * V + o_v
            b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)

            # Scale previous output and add new contribution
            b_o = b_o * exp_diff + exp_s * b_v

            m_max = m_new

    # Normalize output
    b_o = b_o / l_sum

    # Store output
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    p_o = o_ptr + i_b * T * H * V + i_t * H * V + i_h * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)

    # Store LSE for later top-k selection
    b_lse = m_max + tl.log(l_sum)
    p_lse = lse_ptr + i_b * T * H + i_t * H + i_h
    tl.store(p_lse, b_lse)


# ==============================================================================
# Triton Kernel: Sparse Selected Attention
# ==============================================================================

@triton.jit
def nsa_sparse_attn_kernel(
    # Input/Output pointers
    q_ptr,            # Query: [B, T, H, K]
    k_ptr,            # Keys: [B, T, H, K]
    v_ptr,            # Values: [B, T, H, V]
    block_indices_ptr, # Selected block indices: [B, T, H, S]
    o_ptr,            # Output: [B, T, H, V]
    # Dimensions
    B,                # Batch size
    T,                # Sequence length
    S,                # Number of selected blocks (runtime)
    BS,               # Block size (runtime)
    H: tl.constexpr,  # Number of heads
    K: tl.constexpr,  # Key dimension
    V: tl.constexpr,  # Value dimension
    # Scaling
    scale,
    # Block sizes for tiling
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """
    Sparse attention over selected blocks.

    For each query position t, attends to S selected blocks of size BS:
        K_slc = [K[idx*BS:(idx+1)*BS] for idx in block_indices[t]]
        V_slc = [V[idx*BS:(idx+1)*BS] for idx in block_indices[t]]
        o_t = softmax(q_t @ K_slc^T / sqrt(d)) @ V_slc

    Grid: (T, B * H)
    """
    # Get indices
    i_t = tl.program_id(0)   # Token position
    i_bh = tl.program_id(1)  # Batch * head index

    i_b = i_bh // H
    i_h = i_bh % H

    # Load query vector
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    p_q = q_ptr + i_b * T * H * K + i_t * H * K + i_h * K + o_k
    b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32) * scale

    # Load block indices for this position
    p_idx_base = block_indices_ptr + i_b * T * H * S + i_t * H * S + i_h * S

    # Two-pass approach for numerical stability
    # First pass: find maximum score
    m_max = -1e9

    for i_s in range(S):
        block_idx = tl.load(p_idx_base + i_s).to(tl.int32)
        if block_idx < 0:
            pass  # Skip invalid blocks
        else:
            for i_bs in range(BS):
                pos = block_idx * BS + i_bs
                if pos <= i_t and pos < T:
                    p_k = k_ptr + i_b * T * H * K + pos * H * K + i_h * K + o_k
                    b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
                    s = tl.sum(b_q * b_k, axis=0)
                    m_max = tl.maximum(m_max, s)

    # Second pass: compute softmax weighted sum
    l_sum = 0.0
    b_o = tl.zeros([BV], dtype=tl.float32)
    o_v = tl.arange(0, BV)
    m_v = o_v < V

    for i_s in range(S):
        block_idx = tl.load(p_idx_base + i_s).to(tl.int32)
        if block_idx < 0:
            pass
        else:
            for i_bs in range(BS):
                pos = block_idx * BS + i_bs
                if pos <= i_t and pos < T:
                    p_k = k_ptr + i_b * T * H * K + pos * H * K + i_h * K + o_k
                    b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
                    s = tl.sum(b_q * b_k, axis=0)
                    exp_s = tl.exp(s - m_max)
                    l_sum = l_sum + exp_s

                    p_v = v_ptr + i_b * T * H * V + pos * H * V + i_h * V + o_v
                    b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
                    b_o = b_o + exp_s * b_v

    # Normalize output
    l_sum = tl.where(l_sum == 0.0, 1.0, l_sum)
    b_o = b_o / l_sum

    # Store output
    p_o = o_ptr + i_b * T * H * V + i_t * H * V + i_h * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)


# ==============================================================================
# Python Wrappers
# ==============================================================================

def mean_pool_kv(
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean pool keys and values into blocks."""
    B, T, H, K = k.shape
    V = v.shape[-1]

    # Pad to multiple of block_size
    C = (T + block_size - 1) // block_size
    pad_len = C * block_size - T

    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))

    # Reshape and mean
    k_cmp = k.view(B, C, block_size, H, K).mean(dim=2)  # [B, C, H, K]
    v_cmp = v.view(B, C, block_size, H, V).mean(dim=2)  # [B, C, H, V]

    return k_cmp, v_cmp


def nsa_compression_attn(
    q: torch.Tensor,      # [B, T, H, K]
    k_cmp: torch.Tensor,  # [B, C, H, K]
    v_cmp: torch.Tensor,  # [B, C, H, V]
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compression attention: attend to mean-pooled KV blocks.

    Returns:
        o: Output tensor [B, T, H, V]
        lse: Log-sum-exp for block selection [B, T, H]
    """
    B, T, H, K = q.shape
    C = k_cmp.shape[1]
    V = v_cmp.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Block sizes
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)
    BC = 16  # Process 16 compressed blocks at a time

    # Allocate outputs
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)
    lse = torch.empty(B, T, H, dtype=torch.float32, device=q.device)

    # Launch kernel
    grid = (T, B * H)
    nsa_compression_attn_kernel[grid](
        q, k_cmp, v_cmp, o, lse,
        B, T, C, H, K, V, block_size,
        scale,
        BK, BV, BC,
    )

    return o, lse


def select_top_k_blocks(
    q: torch.Tensor,      # [B, T, H, K]
    k_cmp: torch.Tensor,  # [B, C, H, K]
    lse: torch.Tensor,    # [B, T, H]
    block_size: int,
    num_blocks: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Select top-k important blocks based on compression attention scores.

    For each query position, compute importance scores from attention
    and select the top-k blocks.

    Returns:
        block_indices: [B, T, H, S] indices of selected blocks
    """
    B, T, H, K = q.shape
    C = k_cmp.shape[1]

    if scale is None:
        scale = K ** -0.5

    S = min(num_blocks, C)

    # Compute attention scores to all compressed blocks
    q_scaled = q.float() * scale  # [B, T, H, K]
    k_cmp_float = k_cmp.float()   # [B, C, H, K]

    # [B, T, H, C]
    scores = torch.einsum('bthk,bchk->bthc', q_scaled, k_cmp_float)

    # Apply causal mask: only attend to blocks that are complete before position t
    # NC[t] = (t + 1) // block_size
    positions = torch.arange(T, device=q.device)
    num_complete = (positions + 1) // block_size  # [T]
    block_indices_all = torch.arange(C, device=q.device)  # [C]

    # Mask: scores[t, c] valid if c < NC[t]
    causal_mask = block_indices_all[None, :] >= num_complete[:, None]  # [T, C]
    causal_mask = causal_mask[None, :, None, :].expand(B, T, H, C)  # [B, T, H, C]

    # Also give bonus to local block (block containing current position)
    local_block = positions // block_size  # [T]
    local_mask = block_indices_all[None, :] == local_block[:, None]  # [T, C]
    local_mask = local_mask[None, :, None, :].expand(B, T, H, C)  # [B, T, H, C]

    # Convert to attention-like probabilities
    scores_masked = scores.masked_fill(causal_mask, float('-inf'))
    probs = torch.softmax(scores_masked, dim=-1)

    # Give local block max importance (will always be selected)
    probs = probs.masked_fill(local_mask, 1.0)

    # Select top-k blocks
    _, block_indices = probs.topk(S, dim=-1)  # [B, T, H, S]

    # Mark invalid blocks (those beyond causal limit)
    num_complete_expanded = num_complete[None, :, None, None].expand(B, T, H, S)
    invalid_mask = block_indices >= num_complete_expanded
    block_indices = block_indices.masked_fill(invalid_mask, -1)

    return block_indices.to(torch.int32)


def nsa_sparse_attn(
    q: torch.Tensor,            # [B, T, H, K]
    k: torch.Tensor,            # [B, T, H, K]
    v: torch.Tensor,            # [B, T, H, V]
    block_indices: torch.Tensor, # [B, T, H, S]
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Sparse attention over selected blocks.

    Returns:
        o: Output tensor [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = block_indices.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Block sizes
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)

    # Allocate output
    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)

    # Launch kernel
    grid = (T, B * H)
    nsa_sparse_attn_kernel[grid](
        q, k, v, block_indices, o,
        B, T, S, block_size,
        H, K, V,
        scale,
        BK, BV,
    )

    return o


def nsa_sliding_window_attn(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    window_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Sliding window causal attention.

    For simplicity, we use PyTorch implementation here.
    In production, this would use FlashAttention's window mode.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q.float() * scale
    k = k.float()
    v = v.float()

    # [B, H, T, T]
    scores = torch.einsum('bthk,bshk->bhts', q, k)

    # Causal + sliding window mask
    positions = torch.arange(T, device=q.device)
    # Valid if: s <= t and s >= t - window_size + 1
    mask = (positions[None, :] > positions[:, None]) | \
           (positions[:, None] - positions[None, :] >= window_size)
    mask = mask[None, None, :, :].expand(B, H, T, T)

    scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(mask, 0.0)  # Handle all-inf rows

    # [B, H, T, V]
    o = torch.einsum('bhts,bshv->bthv', attn, v)

    return o.to(q.dtype)


def nsa_forward(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g_cmp: torch.Tensor,  # [B, T, H] compression gate
    g_slc: torch.Tensor,  # [B, T, H] selection gate
    g_swa: torch.Tensor,  # [B, T, H] sliding window gate
    block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Full NSA forward pass with all three branches.

    o = g_cmp * o_cmp + g_slc * o_slc + g_swa * o_swa
    """
    B, T, H, K = q.shape

    if scale is None:
        scale = K ** -0.5

    # 1. Compress K, V
    k_cmp, v_cmp = mean_pool_kv(k, v, block_size)

    # 2. Compression attention + get LSE for selection
    o_cmp, lse = nsa_compression_attn(q, k_cmp, v_cmp, block_size, scale)

    # 3. Select important blocks
    block_indices = select_top_k_blocks(q, k_cmp, lse, block_size, num_selected_blocks, scale)

    # 4. Sparse attention over selected blocks
    o_slc = nsa_sparse_attn(q, k, v, block_indices, block_size, scale)

    # 5. Sliding window attention
    o_swa = nsa_sliding_window_attn(q, k, v, window_size, scale)

    # 6. Gated combination
    o = g_cmp.unsqueeze(-1) * o_cmp + \
        g_slc.unsqueeze(-1) * o_slc + \
        g_swa.unsqueeze(-1) * o_swa

    return o


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def nsa_compression_attn_ref(
    q: torch.Tensor,      # [B, T, H, K]
    k_cmp: torch.Tensor,  # [B, C, H, K]
    v_cmp: torch.Tensor,  # [B, C, H, V]
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of compression attention.
    """
    B, T, H, K = q.shape
    C = k_cmp.shape[1]
    V = v_cmp.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q.float() * scale  # [B, T, H, K]
    k_cmp = k_cmp.float()  # [B, C, H, K]
    v_cmp = v_cmp.float()  # [B, C, H, V]

    # Compute attention scores [B, T, H, C]
    scores = torch.einsum('bthk,bchk->bthc', q, k_cmp)

    # Causal mask: only attend to completed blocks
    # NC[t] = (t + 1) // block_size
    positions = torch.arange(T, device=q.device)
    num_complete = (positions + 1) // block_size  # [T]
    block_indices = torch.arange(C, device=q.device)  # [C]

    # Mask: block c is valid for position t if c < NC[t]
    causal_mask = block_indices[None, :] >= num_complete[:, None]  # [T, C]
    causal_mask = causal_mask[None, :, None, :].expand(B, T, H, C)  # [B, T, H, C]

    scores = scores.masked_fill(causal_mask, float('-inf'))

    # Compute softmax and LSE
    lse = torch.logsumexp(scores, dim=-1)  # [B, T, H]
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(causal_mask, 0.0)  # Handle all-inf rows

    # Compute output [B, T, H, V]
    o = torch.einsum('bthc,bchv->bthv', attn, v_cmp)

    # Handle positions with no valid blocks
    no_valid = (num_complete == 0)[None, :, None].expand(B, T, H)
    lse = lse.masked_fill(no_valid, float('-inf'))

    return o.to(q.dtype), lse


def nsa_sparse_attn_ref(
    q: torch.Tensor,            # [B, T, H, K]
    k: torch.Tensor,            # [B, T, H, K]
    v: torch.Tensor,            # [B, T, H, V]
    block_indices: torch.Tensor, # [B, T, H, S]
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of sparse selected attention.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = block_indices.shape[-1]

    if scale is None:
        scale = K ** -0.5

    dtype = q.dtype
    q = q.float() * scale
    k = k.float()
    v = v.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    for b in range(B):
        for t in range(T):
            for h in range(H):
                q_th = q[b, t, h]  # [K]

                # Collect keys and values from selected blocks
                keys = []
                vals = []

                for s in range(S):
                    block_idx = block_indices[b, t, h, s].item()
                    if block_idx < 0:
                        continue

                    start = block_idx * block_size
                    end = min(start + block_size, t + 1)  # Causal: only up to t

                    if start > t:
                        continue

                    for pos in range(start, end):
                        keys.append(k[b, pos, h])
                        vals.append(v[b, pos, h])

                if len(keys) == 0:
                    continue

                keys = torch.stack(keys, dim=0)  # [N, K]
                vals = torch.stack(vals, dim=0)  # [N, V]

                # Attention
                scores = torch.einsum('k,nk->n', q_th, keys)
                attn = torch.softmax(scores, dim=0)
                o[b, t, h] = torch.einsum('n,nv->v', attn, vals)

    return o.to(dtype)


def nsa_forward_ref(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g_cmp: torch.Tensor,  # [B, T, H] compression gate
    g_slc: torch.Tensor,  # [B, T, H] selection gate
    g_swa: torch.Tensor,  # [B, T, H] sliding window gate
    block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation of full NSA.
    """
    B, T, H, K = q.shape

    if scale is None:
        scale = K ** -0.5

    # 1. Compress K, V
    k_cmp, v_cmp = mean_pool_kv(k, v, block_size)

    # 2. Compression attention
    o_cmp, lse = nsa_compression_attn_ref(q, k_cmp, v_cmp, block_size, scale)

    # 3. Select blocks
    block_indices = select_top_k_blocks(q, k_cmp, lse, block_size, num_selected_blocks, scale)

    # 4. Sparse attention
    o_slc = nsa_sparse_attn_ref(q, k, v, block_indices, block_size, scale)

    # 5. Sliding window attention
    o_swa = nsa_sliding_window_attn(q, k, v, window_size, scale)

    # 6. Gated combination
    o = g_cmp.unsqueeze(-1) * o_cmp + \
        g_slc.unsqueeze(-1) * o_slc + \
        g_swa.unsqueeze(-1) * o_swa

    return o


# ==============================================================================
# Helion Implementation
# ==============================================================================

if HELION_AVAILABLE:
    @helion.kernel(static_shapes=True, autotune_effort="none")
    def nsa_compression_attn_helion_kernel(
        q: torch.Tensor,       # [B, T, H, K]
        k_cmp: torch.Tensor,   # [B, C, H, K]
        v_cmp: torch.Tensor,   # [B, C, H, V]
        lse_out: torch.Tensor, # [B, T, H] pre-allocated for LSE output
        block_size: int,
        sm_scale: float,
    ) -> torch.Tensor:
        """
        Helion implementation of NSA compression attention.

        Attends to mean-pooled KV blocks with causal masking:
        For each query position t, only attends to completed blocks:
            NC = (t + 1) // block_size

        Args:
            q: Query tensor [B, T, H, K]
            k_cmp: Compressed keys [B, C, H, K]
            v_cmp: Compressed values [B, C, H, V]
            lse_out: Pre-allocated LSE output [B, T, H]
            block_size: Compression block size
            sm_scale: Softmax scale factor

        Returns:
            Output tensor [B, T, H, V]
        """
        B, T, H, K = q.shape
        C = k_cmp.shape[1]
        V = v_cmp.shape[-1]
        K = hl.specialize(K)
        V = hl.specialize(V)
        block_size = hl.specialize(block_size)

        # Output tensor
        o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)

        # Scale for exp2 instead of exp (1/log(2) = 1.44269504)
        qk_scale = sm_scale * 1.44269504

        # Use large negative value instead of -inf to avoid NaN
        NEG_INF = -1e10

        # Process each batch, head, and query tile
        for tile_b, tile_h, tile_m in hl.tile([B, H, T], block_size=[1, 1, None]):
            # Initialize online softmax accumulators
            m_i = hl.full([tile_m], NEG_INF, dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, V], dtype=torch.float32)

            # Load query tile: [tile_m, K]
            # Note: q is [B, T, H, K], we want q[b, tile_m, h, :]
            q_tile = q[tile_b.begin, tile_m, tile_h.begin, :]

            # Iterate over compressed blocks
            for tile_n in hl.tile(C):
                # Load key tile: [tile_n, K]
                # k_cmp is [B, C, H, K], we want k_cmp[b, tile_n, h, :]
                k_tile = k_cmp[tile_b.begin, tile_n, tile_h.begin, :]
                k_tile_t = k_tile.T  # [K, tile_n]

                # Compute attention scores: [tile_m, K] @ [K, tile_n] = [tile_m, tile_n]
                qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk = hl.dot(q_tile, k_tile_t, acc=qk)

                # Create position indices for masking
                offs_m = tile_m.begin + hl.arange(tile_m.block_size)  # Query positions
                offs_n = tile_n.begin + hl.arange(tile_n.block_size)  # Block indices

                # Causal mask for compression attention:
                # For query position t, block c is valid if c < (t + 1) // block_size
                # Equivalent to: (c + 1) * block_size <= t + 1
                # This avoids integer division which may have issues
                valid_mask = (offs_n[None, :] + 1) * block_size <= (offs_m[:, None] + 1)

                # Also need bounds check for valid block indices
                valid_mask = valid_mask & (offs_n[None, :] < C)

                # Scale QK first, then apply mask for numerical stability
                qk = qk * qk_scale

                # Apply mask BEFORE computing max to get correct m_ij
                qk = torch.where(valid_mask, qk, NEG_INF)

                # Online softmax update
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                qk = qk - m_ij[:, None]
                p = torch.exp2(qk)

                # Apply mask AFTER exp2 to zero out invalid positions
                # This prevents masked positions from contributing to the sum
                # when all positions are masked (NEG_INF - NEG_INF = 0, exp2(0) = 1)
                p = torch.where(valid_mask, p, 0.0)

                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]

                # Load values and accumulate
                # v_cmp is [B, C, H, V], we want v_cmp[b, tile_n, h, :]
                v_tile = v_cmp[tile_b.begin, tile_n, tile_h.begin, :]  # [tile_n, V]
                p = p.to(v_tile.dtype)
                acc = hl.dot(p, v_tile, acc=acc)

                m_i = m_ij

            # Normalize output - handle case where l_i is very small
            l_i = torch.maximum(l_i, torch.full_like(l_i, 1e-10))
            acc = acc / l_i[:, None]

            # Store output
            o[tile_b, tile_m, tile_h, :] = acc[None, :, None, :].to(o.dtype)

            # Store LSE: convert from base-2 to base-e
            # m_i is max of (scores * qk_scale) where qk_scale = sm_scale * LOG2_E
            # To get base-e LSE: lse = m_i / LOG2_E + log(l_i)
            LOG2_E = 1.44269504
            lse_val = m_i / LOG2_E + torch.log(l_i)
            lse_out[tile_b, tile_m, tile_h] = lse_val[None, :, None]

        return o


def nsa_compression_attn_helion(
    q: torch.Tensor,      # [B, T, H, K]
    k_cmp: torch.Tensor,  # [B, C, H, K]
    v_cmp: torch.Tensor,  # [B, C, H, V]
    block_size: int,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helion implementation of compression attention.

    Returns:
        o: Output tensor [B, T, H, V]
        lse: Log-sum-exp for block selection [B, T, H]
    """
    if not HELION_AVAILABLE:
        raise RuntimeError("Helion is not available")

    B, T, H, K = q.shape
    V = v_cmp.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Pre-allocate LSE output
    lse_out = torch.empty(B, T, H, dtype=torch.float32, device=q.device)

    # Call Helion kernel
    o = nsa_compression_attn_helion_kernel(q, k_cmp, v_cmp, lse_out, block_size, scale)

    return o, lse_out


# ==============================================================================
# Numerical Tests
# ==============================================================================

def test_nsa_compression_attn():
    """Test compression attention kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("Testing NSA compression attention...")

    configs = [
        # (B, T, H, K, V, BS)
        (1, 64, 2, 32, 32, 16),
        (2, 128, 4, 64, 64, 32),
        (2, 256, 4, 64, 64, 64),
    ]

    for B, T, H, K, V, BS in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}")

        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)

        # Compress
        k_cmp, v_cmp = mean_pool_kv(k, v, BS)

        # Reference
        ref_o, ref_lse = nsa_compression_attn_ref(q, k_cmp, v_cmp, BS)

        # Triton
        tri_o, tri_lse = nsa_compression_attn(q, k_cmp, v_cmp, BS)

        # Check
        o_atol = (ref_o - tri_o).abs().max().item()
        lse_atol = (ref_lse - tri_lse).abs().max().item()

        # Filter out -inf comparisons for LSE
        valid_lse = ~(ref_lse.isinf() | tri_lse.isinf())
        if valid_lse.any():
            lse_atol = (ref_lse[valid_lse] - tri_lse[valid_lse]).abs().max().item()
        else:
            lse_atol = 0.0

        status = "PASS" if o_atol < 1e-3 and lse_atol < 1e-3 else "FAIL"
        print(f"    Output atol={o_atol:.2e}, LSE atol={lse_atol:.2e} [{status}]")

        if status == "FAIL":
            raise AssertionError(f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}")

    print("  All compression attention tests passed!")


def test_nsa_sparse_attn():
    """Test sparse selected attention kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("\nTesting NSA sparse selected attention...")

    configs = [
        # (B, T, H, K, V, BS, S)
        (1, 64, 2, 32, 32, 16, 4),
        (2, 128, 2, 64, 64, 32, 4),
        (1, 256, 4, 64, 64, 64, 4),
    ]

    for B, T, H, K, V, BS, S in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}, S={S}")

        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)

        # Generate valid block indices
        C = (T + BS - 1) // BS
        block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=device)
        for t in range(T):
            num_complete = (t + 1) // BS
            if num_complete > 0:
                available = min(num_complete, S)
                for s in range(available):
                    block_indices[:, t, :, s] = s
                for s in range(available, S):
                    block_indices[:, t, :, s] = -1
            else:
                block_indices[:, t, :, :] = -1

        # Reference
        ref_o = nsa_sparse_attn_ref(q, k, v, block_indices, BS)

        # Triton
        tri_o = nsa_sparse_attn(q, k, v, block_indices, BS)

        # Check
        atol = (ref_o - tri_o).abs().max().item()
        rtol = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

        status = "PASS" if atol < 1e-3 or rtol < 1e-2 else "FAIL"
        print(f"    atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

        if status == "FAIL":
            raise AssertionError(f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}, S={S}")

    print("  All sparse attention tests passed!")


def test_nsa_full_forward():
    """Test full NSA forward pass."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("\nTesting full NSA forward pass...")

    B, T, H, K, V = 1, 128, 2, 64, 64
    BS = 32
    S = 4
    W = 32

    print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}, S={S}, W={W}")

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)

    # Gates (typically sigmoid outputs)
    g_cmp = torch.sigmoid(torch.randn(B, T, H, device=device))
    g_slc = torch.sigmoid(torch.randn(B, T, H, device=device))
    g_swa = torch.sigmoid(torch.randn(B, T, H, device=device))

    # Reference
    ref_o = nsa_forward_ref(q, k, v, g_cmp, g_slc, g_swa, BS, S, W)

    # Triton
    tri_o = nsa_forward(q, k, v, g_cmp, g_slc, g_swa, BS, S, W)

    # Check
    atol = (ref_o - tri_o).abs().max().item()
    rtol = ((ref_o - tri_o).abs() / (ref_o.abs() + 1e-8)).max().item()

    status = "PASS" if atol < 1e-2 or rtol < 5e-2 else "FAIL"
    print(f"    atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

    if status == "FAIL":
        raise AssertionError("Full NSA forward test failed")

    print("  Full NSA forward test passed!")


def test_nsa_block_selection():
    """Test block selection produces valid indices."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("\nTesting NSA block selection...")

    B, T, H, K = 2, 128, 4, 64
    BS = 32
    S = 4

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, K, dtype=torch.float32, device=device)

    k_cmp, v_cmp = mean_pool_kv(k, v, BS)
    _, lse = nsa_compression_attn_ref(q, k_cmp, v_cmp, BS)

    block_indices = select_top_k_blocks(q, k_cmp, lse, BS, S)

    # Verify indices are valid
    C = k_cmp.shape[1]

    for t in range(T):
        NC = (t + 1) // BS
        for b in range(B):
            for h in range(H):
                for s in range(S):
                    idx = block_indices[b, t, h, s].item()
                    # Either -1 (invalid) or in valid range
                    valid = (idx == -1) or (0 <= idx < NC)
                    if not valid:
                        raise AssertionError(
                            f"Invalid block index {idx} at t={t}, NC={NC}"
                        )

    print("  Block selection produces valid indices!")
    print("  PASS")


def test_nsa_compression_attn_helion():
    """Test Helion implementation of compression attention."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    if not HELION_AVAILABLE:
        print("Helion not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("\nTesting NSA compression attention (Helion)...")

    configs = [
        # (B, T, H, K, V, BS)
        (1, 64, 2, 32, 32, 16),
        (2, 128, 4, 64, 64, 32),
        (2, 256, 4, 64, 64, 64),
    ]

    for B, T, H, K, V, BS in configs:
        print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}")

        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)

        # Compress
        k_cmp, v_cmp = mean_pool_kv(k, v, BS)

        # Reference
        ref_o, ref_lse = nsa_compression_attn_ref(q, k_cmp, v_cmp, BS)

        # Helion
        helion_o, helion_lse = nsa_compression_attn_helion(q, k_cmp, v_cmp, BS)

        # Check output
        o_atol = (ref_o - helion_o).abs().max().item()

        # Filter out -inf comparisons for LSE
        valid_lse = ~(ref_lse.isinf() | helion_lse.isinf())
        if valid_lse.any():
            lse_atol = (ref_lse[valid_lse] - helion_lse[valid_lse]).abs().max().item()
        else:
            lse_atol = 0.0

        status = "PASS" if o_atol < 1e-3 and lse_atol < 1e-3 else "FAIL"
        print(f"    Output atol={o_atol:.2e}, LSE atol={lse_atol:.2e} [{status}]")

        if status == "FAIL":
            raise AssertionError(f"Test failed for config B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}")

    print("  All Helion compression attention tests passed!")


def benchmark_nsa():
    """Simple benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    import time

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("\n" + "=" * 60)
    print("NSA Benchmark")
    print("=" * 60)

    B, T, H, K, V = 2, 512, 4, 64, 64
    BS = 64
    S = 8
    W = 64

    print(f"Config: B={B}, T={T}, H={H}, K={K}, V={V}, BS={BS}, S={S}, W={W}")

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g_cmp = torch.sigmoid(torch.randn(B, T, H, device=device))
    g_slc = torch.sigmoid(torch.randn(B, T, H, device=device))
    g_swa = torch.sigmoid(torch.randn(B, T, H, device=device))

    # Warmup
    for _ in range(3):
        nsa_forward(q, k, v, g_cmp, g_slc, g_swa, BS, S, W)
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        nsa_forward(q, k, v, g_cmp, g_slc, g_swa, BS, S, W)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / n_iters * 1000

    print(f"Triton NSA: {triton_time:.2f} ms")

    # Compare to standard attention FLOPs
    standard_attn_flops = B * H * T * T * K * 2  # Approximate
    nsa_effective_tokens = (T // BS) + S * BS + W  # Compressed + selected + window
    nsa_approx_flops = B * H * T * nsa_effective_tokens * K * 2

    print(f"Standard attention equivalent tokens: {T}")
    print(f"NSA effective tokens per query: ~{nsa_effective_tokens}")
    print(f"Theoretical speedup: ~{T / nsa_effective_tokens:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("NSA (Native Sparse Attention) Triton Core Tests")
    print("=" * 60)

    test_nsa_compression_attn()
    test_nsa_sparse_attn()
    test_nsa_full_forward()
    test_nsa_block_selection()
    test_nsa_compression_attn_helion()
    benchmark_nsa()

    print("\n" + "=" * 60)
    print("All NSA tests completed successfully!")
    print("=" * 60)
