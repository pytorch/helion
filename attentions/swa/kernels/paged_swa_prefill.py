# Paged Sliding Window Attention (SWA) Prefill Kernel
#
# This kernel implements paged prefill attention with sliding window support.
# Combines:
# - Flash Attention 2 style tiled computation
# - CSR-style paged KV cache access
# - Tile-level pruning for sliding window (skips tiles outside window)
#
# Key optimization: For sliding window, we compute which KV tiles can
# possibly contribute to each query tile, and skip the rest entirely.

import torch
import triton
import triton.language as tl


def is_cuda():
    """Check if running on CUDA."""
    return torch.cuda.is_available()


_is_cuda = is_cuda()

if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


@triton.jit
def _paged_swa_prefill_kernel(
    Q,
    K,
    V,
    O,
    b_start_loc,
    b_seq_len,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    sliding_window: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Paged prefill attention with sliding window.

    For each query block, we compute the valid KV range considering:
    1. Causal mask: KV position <= query position
    2. Sliding window: KV position >= query position - window_size

    The valid range is: [max(0, q_pos - W), q_pos] for causal SWA
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Get sequence info
    cur_batch_start = tl.load(b_start_loc + cur_batch)
    cur_batch_seq_len = tl.load(b_seq_len + cur_batch)

    # Check if this block is valid
    block_start_m = cur_block_m * BLOCK_M
    if block_start_m >= cur_batch_seq_len:
        return

    # Offsets for Q block
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # Pointers
    q_ptrs = (
        Q
        + (cur_batch_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    k_ptrs = (
        K
        + cur_batch_start * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + cur_batch_start * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :]
    )

    # Load Q block
    q_mask = offs_m[:, None] < cur_batch_seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Compute KV range for this Q block
    # For query positions [block_start_m, block_start_m + BLOCK_M)
    # The valid KV range depends on causal + sliding window constraints

    if IS_CAUSAL:
        # Upper bound: last query position in this block
        kv_end = tl.minimum(block_start_m + BLOCK_M, cur_batch_seq_len)
    else:
        kv_end = cur_batch_seq_len

    if sliding_window > 0:
        # Lower bound: first query position minus window size
        kv_start = tl.maximum(0, block_start_m - sliding_window + 1)
    else:
        kv_start = 0

    # Round kv_start down to block boundary for efficient iteration
    kv_start_block = (kv_start // BLOCK_N) * BLOCK_N

    # Iterate over KV blocks
    for start_n in range(kv_start_block, kv_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_offs_n = start_n + offs_n

        # Load K block
        k_block_ptrs = k_ptrs + cur_offs_n[:, None] * stride_kbs
        k_mask = cur_offs_n[:, None] < cur_batch_seq_len
        k = tl.load(k_block_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Apply masks
        # 1. Sequence length mask
        qk = tl.where(
            (offs_m[:, None] < cur_batch_seq_len) & (cur_offs_n[None, :] < cur_batch_seq_len),
            qk,
            float("-inf"),
        )

        # 2. Causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # 3. Sliding window mask
        if sliding_window > 0:
            # Valid: kv_pos >= q_pos - window_size + 1
            window_mask = cur_offs_n[None, :] >= (offs_m[:, None] - sliding_window + 1)
            qk = tl.where(window_mask, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, 1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)
        p = tl.exp(qk - new_m_i[:, None])

        # Load V block
        v_block_ptrs = v_ptrs + cur_offs_n[:, None] * stride_vbs
        v = tl.load(v_block_ptrs, mask=k_mask, other=0.0)

        # Update accumulators
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = new_m_i

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = (
        O
        + (cur_batch_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    o_mask = offs_m[:, None] < cur_batch_seq_len
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)


def paged_swa_prefill_fwd(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    b_start_loc: torch.Tensor,  # [batch_size]
    b_seq_len: torch.Tensor,  # [batch_size]
    max_input_len: int,
    sliding_window: int,
    is_causal: bool = True,
):
    """
    Paged prefill attention with sliding window.

    This kernel handles variable-length sequences packed into a single tensor,
    with CSR-style indexing (b_start_loc gives the start of each sequence).

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim]
        k: Key tensor [total_tokens, num_kv_heads, head_dim]
        v: Value tensor [total_tokens, num_kv_heads, head_dim]
        o: Output tensor [total_tokens, num_heads, head_dim]
        b_start_loc: Start locations [batch_size]
        b_seq_len: Sequence lengths [batch_size]
        max_input_len: Maximum sequence length
        sliding_window: Window size (tokens), 0 or negative means no window
        is_causal: Whether to apply causal masking
    """
    head_dim = q.shape[-1]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    batch_size = b_seq_len.shape[0]
    kv_group_num = num_heads // num_kv_heads

    # Block sizes
    if _is_cuda and CUDA_CAPABILITY[0] >= 9:
        BLOCK_M, BLOCK_N = 128, 64
        num_warps = 8
    elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
        BLOCK_M, BLOCK_N = 64, 64
        num_warps = 4
    else:
        BLOCK_M, BLOCK_N = 64, 64
        num_warps = 4

    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    sm_scale = 1.0 / (head_dim ** 0.5)

    grid = (batch_size, num_heads, triton.cdiv(max_input_len, BLOCK_M))

    _paged_swa_prefill_kernel[grid](
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        sliding_window=sliding_window if sliding_window > 0 else 0,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=2,
    )


# ============================================================================
# Extend Attention with Sliding Window (Prefix + Extend)
# ============================================================================


@triton.jit
def _paged_swa_extend_kernel(
    Q_extend,
    K_extend,
    V_extend,
    O_extend,
    K_buffer,
    V_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    sliding_window: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Extend attention with sliding window and paged KV cache.

    This handles the case where we have:
    - Prefix KV in a paged buffer (k_buffer, v_buffer via kv_indices)
    - Extend KV that's contiguous (k_extend, v_extend)

    For sliding window, we may be able to skip prefix entirely if
    window_size < extend_len.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Get sequence boundaries
    q_start = tl.load(qo_indptr + cur_batch)
    q_end = tl.load(qo_indptr + cur_batch + 1)
    extend_len = q_end - q_start

    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    prefix_len = kv_end - kv_start

    total_kv_len = prefix_len + extend_len

    # Check if this block is valid
    block_start_m = cur_block_m * BLOCK_M
    if block_start_m >= extend_len:
        return

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q from extend
    q_ptrs = (
        Q_extend
        + (q_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q_mask = offs_m[:, None] < extend_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Compute valid KV range considering sliding window
    # Query positions are: [block_start_m, block_start_m + BLOCK_M) relative to extend
    # Actual positions: prefix_len + query_position_in_extend

    if sliding_window > 0:
        # First query in block is at global position: prefix_len + block_start_m
        # Valid KV starts at: prefix_len + block_start_m - window_size + 1
        kv_lower = tl.maximum(0, prefix_len + block_start_m - sliding_window + 1)
    else:
        kv_lower = 0

    if IS_CAUSAL:
        # Last query in block is at: prefix_len + min(block_start_m + BLOCK_M, extend_len) - 1
        # Valid KV ends at: that position + 1
        kv_upper = prefix_len + tl.minimum(block_start_m + BLOCK_M, extend_len)
    else:
        kv_upper = total_kv_len

    # Process prefix KV (if any falls in valid range)
    prefix_start = tl.maximum(0, kv_lower)
    prefix_end = tl.minimum(prefix_len, kv_upper)

    if prefix_end > prefix_start:
        prefix_start_block = (prefix_start // BLOCK_N) * BLOCK_N

        for start_n in range(prefix_start_block, prefix_end, BLOCK_N):
            cur_offs_n = start_n + offs_n

            # Load K from prefix buffer
            kv_idx = tl.load(
                kv_indices + kv_start + cur_offs_n,
                mask=cur_offs_n < prefix_len,
                other=0,
            )
            k_ptrs = (
                kv_idx[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(K_buffer + k_ptrs, mask=(cur_offs_n[:, None] < prefix_len), other=0.0)

            # Compute QK
            qk = tl.dot(q, tl.trans(k))
            qk *= sm_scale

            # Apply masks
            qk = tl.where(cur_offs_n[None, :] < prefix_len, qk, float("-inf"))

            if IS_CAUSAL:
                # Global query pos: prefix_len + offs_m
                # Global kv pos: cur_offs_n
                causal_mask = (prefix_len + offs_m[:, None]) >= cur_offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            if sliding_window > 0:
                window_mask = cur_offs_n[None, :] >= (prefix_len + offs_m[:, None] - sliding_window + 1)
                qk = tl.where(window_mask, qk, float("-inf"))

            # Load V from prefix buffer
            v_ptrs = (
                kv_idx[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_d[None, :]
            )
            v = tl.load(V_buffer + v_ptrs, mask=(cur_offs_n[:, None] < prefix_len), other=0.0)

            # Online softmax update
            m_ij = tl.max(qk, 1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - new_m_i)
            p = tl.exp(qk - new_m_i[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = new_m_i

    # Process extend KV
    extend_kv_start = tl.maximum(0, kv_lower - prefix_len)
    extend_kv_end = tl.minimum(extend_len, kv_upper - prefix_len)

    if extend_kv_end > extend_kv_start:
        extend_start_block = (extend_kv_start // BLOCK_N) * BLOCK_N

        for start_n in range(extend_start_block, extend_kv_end, BLOCK_N):
            cur_offs_n = start_n + offs_n  # Position in extend

            # Load K from extend
            k_ptrs = (
                K_extend
                + (q_start + cur_offs_n[:, None]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=(cur_offs_n[:, None] < extend_len), other=0.0)

            # Compute QK
            qk = tl.dot(q, tl.trans(k))
            qk *= sm_scale

            # Apply masks
            qk = tl.where(cur_offs_n[None, :] < extend_len, qk, float("-inf"))

            if IS_CAUSAL:
                # Query pos in extend: offs_m
                # KV pos in extend: cur_offs_n
                causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            if sliding_window > 0:
                # Global query pos: prefix_len + offs_m
                # Global kv pos: prefix_len + cur_offs_n
                # Window mask: kv >= query - window + 1
                window_mask = (prefix_len + cur_offs_n[None, :]) >= (prefix_len + offs_m[:, None] - sliding_window + 1)
                qk = tl.where(window_mask, qk, float("-inf"))

            # Load V from extend
            v_ptrs = (
                V_extend
                + (q_start + cur_offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=(cur_offs_n[:, None] < extend_len), other=0.0)

            # Online softmax update
            m_ij = tl.max(qk, 1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - new_m_i)
            p = tl.exp(qk - new_m_i[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = new_m_i

    # Normalize and store
    acc = acc / l_i[:, None]
    o_ptrs = (
        O_extend
        + (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    o_mask = offs_m[:, None] < extend_len
    tl.store(o_ptrs, acc.to(O_extend.dtype.element_ty), mask=o_mask)


def paged_swa_extend_fwd(
    q_extend: torch.Tensor,  # [total_extend_tokens, num_heads, head_dim]
    k_extend: torch.Tensor,  # [total_extend_tokens, num_kv_heads, head_dim]
    v_extend: torch.Tensor,  # [total_extend_tokens, num_kv_heads, head_dim]
    o_extend: torch.Tensor,  # [total_extend_tokens, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_prefix_tokens, num_kv_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_prefix_tokens, num_kv_heads, head_dim]
    qo_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # [total_prefix_tokens]
    max_extend_len: int,
    sliding_window: int,
    is_causal: bool = True,
):
    """
    Extend attention with sliding window and paged KV cache.

    Args:
        q_extend: New query tokens
        k_extend: New key tokens
        v_extend: New value tokens
        o_extend: Output for new tokens
        k_buffer: Prefix K cache (paged)
        v_buffer: Prefix V cache (paged)
        qo_indptr: Query/output boundaries [batch_size + 1]
        kv_indptr: Prefix KV boundaries [batch_size + 1]
        kv_indices: Physical indices for prefix KV
        max_extend_len: Maximum extend length
        sliding_window: Window size (tokens), 0 or negative means no window
        is_causal: Whether to apply causal masking
    """
    head_dim = q_extend.shape[-1]
    num_heads = q_extend.shape[1]
    num_kv_heads = k_extend.shape[1]
    batch_size = qo_indptr.shape[0] - 1
    kv_group_num = num_heads // num_kv_heads

    # Block sizes
    BLOCK_M, BLOCK_N = 64, 64
    num_warps = 4

    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    sm_scale = 1.0 / (head_dim ** 0.5)

    grid = (batch_size, num_heads, triton.cdiv(max_extend_len, BLOCK_M))

    _paged_swa_extend_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        sliding_window=sliding_window if sliding_window > 0 else 0,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=2,
    )
