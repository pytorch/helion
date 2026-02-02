# Paged Sliding Window Attention (SWA) Decode Kernel
#
# This kernel implements paged decode attention with sliding window support.
# Key optimization: Tile-level pruning that skips entire KV blocks outside
# the sliding window, providing O(W) instead of O(N) complexity.
#
# Supports both CSR-style (kv_indptr/kv_indices) and block table indexing.

import torch
import triton
import triton.language as tl




@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _paged_swa_decode_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    sliding_window: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Stage 1: Parallel partial attention over KV splits with sliding window.

    The key optimization is tile-level pruning:
    - For each query at position Q, only KV positions in [Q - W, Q] are valid
    - We skip entire tiles that fall outside this window
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    # Get KV range for this batch
    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # Exit if this split is not needed
    if split_kv_id >= kv_splits:
        return

    # Calculate the KV range for this split
    kv_per_split = tl.cdiv(cur_batch_seq_len, kv_splits)
    split_kv_start = split_kv_id * kv_per_split
    split_kv_end = tl.minimum(split_kv_start + kv_per_split, cur_batch_seq_len)

    # SLIDING WINDOW OPTIMIZATION: Restrict to valid window
    # For decode, query is at position (seq_len - 1), so valid KV is in [seq_len - W, seq_len - 1]
    if sliding_window > 0:
        window_start = tl.maximum(0, cur_batch_seq_len - sliding_window)
        # Clamp the split range to the window
        split_kv_start = tl.maximum(split_kv_start, window_start)
        if split_kv_start >= split_kv_end:
            # This entire split is outside the window, skip it
            # Write zeros to output
            offs_mid_o = (
                cur_batch * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_dv
            )
            tl.store(Att_Out + offs_mid_o, tl.zeros([BLOCK_DV], dtype=tl.float32), mask=mask_dv)
            offs_mid_lse = (
                cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os
            ) // Lv
            tl.store(Att_Lse + offs_mid_lse, float("-inf"))
            return

    # Load query
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + offs_q, mask=mask_d, other=0.0)

    # Initialize accumulators (using float instead of [1] array)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # Iterate over KV blocks in this split
    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_loc = tl.load(
            kv_indices + cur_batch_kv_start_idx + offs_n,
            mask=offs_n < split_kv_end,
            other=0,
        )

        # Load K
        offs_buf_k = (
            kv_loc[:, None] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[None, :]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
            other=0.0,
        )

        # Compute QK
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
        qk *= sm_scale

        # Apply sliding window mask within the block
        if sliding_window > 0:
            # Mask out positions outside window
            kv_positions = offs_n
            # For decode, query is at seq_len - 1
            valid_mask = kv_positions >= (cur_batch_seq_len - sliding_window)
            qk = tl.where(valid_mask & (offs_n < split_kv_end), qk, float("-inf"))
        else:
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

        # Load V
        offs_buf_v = (
            kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v,
            mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
            other=0.0,
        )

        # Online softmax update
        block_max = tl.max(qk, 0)
        n_e_max = tl.maximum(block_max, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * re_scale + tl.sum(p[:, None] * v, axis=0)
        p_sum = tl.sum(p, 0)
        e_sum = e_sum * re_scale + p_sum
        e_max = n_e_max

    # Store partial results
    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_dv
    )
    # Store UNnormalized acc (important for correct log-sum-exp reduction in stage2)
    # acc already contains: sum_t(exp(qk_t - e_max) * v_t)
    # We need to store it scaled by exp(e_max) to get: sum_t(exp(qk_t) * v_t) / e_sum
    # Actually, for the log-sum-exp reduction, we store acc / e_sum (normalized)
    # but also need to track the total mass via lse = e_max + log(e_sum) = log(Z)
    # Then in stage2: acc_combined = sum_i(acc_i * Z_i) / Z_total
    #                              = sum_i(acc_i * exp(lse_i)) / exp(lse_total)
    # Since acc_i is normalized, acc_i * Z_i = unnormalized sum
    acc_normalized = acc / e_sum
    tl.store(Att_Out + offs_mid_o, acc_normalized, mask=mask_dv)

    # Store LSE as a scalar: lse = log(Z) = e_max + log(e_sum)
    offs_mid_lse = (
        cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os
    ) // Lv
    lse_val = e_max + tl.log(e_sum)
    tl.store(Att_Lse + offs_mid_lse, lse_val)


@triton.jit
def _paged_swa_decode_stage2(
    Mid_O,
    Mid_Lse,
    O,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """Stage 2: Reduce partial results across splits."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    if cur_batch_seq_len == 0:
        return

    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    # Initialize with first split
    offs_mid = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + offs_dv
    )
    offs_lse = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv

    acc = tl.load(Mid_O + offs_mid, mask=mask_dv, other=0.0)
    lse = tl.load(Mid_Lse + offs_lse)

    # Reduce across splits (use conditional instead of break)
    for split_id in range(1, MAX_KV_SPLITS):
        if split_id < kv_splits:
            offs_mid_i = offs_mid + split_id * stride_mid_os
            offs_lse_i = offs_lse + split_id * stride_mid_os // Lv

            acc_i = tl.load(Mid_O + offs_mid_i, mask=mask_dv, other=0.0)
            lse_i = tl.load(Mid_Lse + offs_lse_i)

            # Log-sum-exp reduction with proper handling of -inf
            # The math: we want acc_new such that acc_new is the normalized weighted combination
            # acc_new = (acc * Z + acc_i * Z_i) / (Z + Z_i)
            # where Z = exp(lse) and Z_i = exp(lse_i)
            new_lse = tl.maximum(lse, lse_i)
            # Handle -inf case: when lse is -inf, exp(lse - new_lse) should be 0, not nan
            is_valid_lse = lse > float("-inf")
            is_valid_lse_i = lse_i > float("-inf")
            scale = tl.where(is_valid_lse, tl.exp(lse - new_lse), 0.0)
            scale_i = tl.where(is_valid_lse_i, tl.exp(lse_i - new_lse), 0.0)
            total_scale = scale + scale_i
            # Normalize the combined acc: acc = (acc*scale + acc_i*scale_i) / (scale + scale_i)
            # This gives the correctly weighted average
            acc = tl.where(total_scale > 0, (acc * scale + acc_i * scale_i) / total_scale, 0.0)
            # Update lse (this tracks log(Z_total))
            lse = tl.where(is_valid_lse | is_valid_lse_i, new_lse + tl.log(total_scale), float("-inf"))

    # Normalize and store
    offs_o = cur_batch * stride_obs + cur_head * stride_oh + offs_dv
    tl.store(O + offs_o, acc, mask=mask_dv)


def paged_swa_decode_fwd(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_heads, head_dim]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # [total_kv_tokens]
    attn_logits: torch.Tensor,  # [batch_size, num_heads, max_kv_splits, head_dim]
    attn_lse: torch.Tensor,  # [batch_size, num_heads, max_kv_splits]
    num_kv_splits: torch.Tensor,  # [batch_size]
    max_kv_splits: int,
    sm_scale: float,
    sliding_window: int,
):
    """
    Paged decode attention with sliding window.

    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_buffer: Paged K cache [total_kv_tokens, num_kv_heads, head_dim]
        v_buffer: Paged V cache [total_kv_tokens, num_kv_heads, head_dim]
        o: Output tensor [batch_size, num_heads, head_dim]
        kv_indptr: CSR-style pointers [batch_size + 1]
        kv_indices: Physical KV indices [total_kv_tokens]
        attn_logits: Intermediate buffer for partial outputs
        attn_lse: Intermediate buffer for log-sum-exp values
        num_kv_splits: Number of splits per sequence
        max_kv_splits: Maximum number of splits
        sm_scale: Softmax scale (typically 1/sqrt(head_dim))
        sliding_window: Window size (tokens), 0 or negative means no window
    """
    BLOCK_N = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch_size, num_heads = q.shape[0], q.shape[1]
    kv_group_num = num_heads // k_buffer.shape[1]

    # Stage 1: Parallel partial attention
    grid_stage1 = (batch_size, num_heads, max_kv_splits)

    num_stages = 2

    _paged_swa_decode_stage1[grid_stage1](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        sliding_window=sliding_window if sliding_window > 0 else 0,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        Lk=Lk,
        Lv=Lv,
        num_warps=4,
        num_stages=num_stages,
    )

    # Stage 2: Reduce across splits
    grid_stage2 = (batch_size, num_heads)

    _paged_swa_decode_stage2[grid_stage2](
        attn_logits,
        attn_lse,
        o,
        kv_indptr,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


# ============================================================================
# Block Table Version
# ============================================================================


@triton.jit
def _paged_swa_decode_blocktable_kernel(
    Q,
    K_Cache,
    V_Cache,
    O,
    block_table,
    seq_lens,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kc_blk,
    stride_kc_tok,
    stride_kc_h,
    stride_vc_blk,
    stride_vc_tok,
    stride_vc_h,
    stride_obs,
    stride_oh,
    stride_bt,
    sliding_window: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Paged decode with block table indexing and sliding window.

    KV Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    seq_len = tl.load(seq_lens + cur_batch)
    if seq_len == 0:
        return

    offs_d = tl.arange(0, HEAD_DIM)

    # Load query
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + offs_q)

    # Compute window bounds
    if sliding_window > 0:
        window_start = tl.maximum(0, seq_len - sliding_window)
    else:
        window_start = 0
    window_end = seq_len

    # Initialize accumulators
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Iterate over KV positions in window
    for kv_pos in range(window_start, window_end, BLOCK_N):
        offs_n = kv_pos + tl.arange(0, BLOCK_N)
        valid_mask = offs_n < window_end

        # Compute block and offset for each position
        block_idx = offs_n // BLOCK_SIZE
        block_offset = offs_n % BLOCK_SIZE

        # Load physical block indices from block table
        physical_blocks = tl.load(
            block_table + cur_batch * stride_bt + block_idx,
            mask=valid_mask,
            other=0,
        )

        # Load K values
        k_ptrs = (
            physical_blocks[:, None] * stride_kc_blk
            + block_offset[:, None] * stride_kc_tok
            + cur_kv_head * stride_kc_h
            + offs_d[None, :]
        )
        k = tl.load(K_Cache + k_ptrs, mask=valid_mask[:, None], other=0.0)

        # Compute attention scores
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_N]
        qk = qk * sm_scale
        qk = tl.where(valid_mask, qk, float("-inf"))

        # Load V values
        v_ptrs = (
            physical_blocks[:, None] * stride_vc_blk
            + block_offset[:, None] * stride_vc_tok
            + cur_kv_head * stride_vc_h
            + offs_d[None, :]
        )
        v = tl.load(V_Cache + v_ptrs, mask=valid_mask[:, None], other=0.0)

        # Online softmax
        m_ij = tl.max(qk, 0)
        new_m = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(qk - new_m)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = new_m

    # Normalize and store
    acc = acc / l_i
    offs_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(O + offs_o, acc)


def paged_swa_decode_blocktable_fwd(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    o: torch.Tensor,  # [batch_size, num_heads, head_dim]
    block_table: torch.Tensor,  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    sm_scale: float,
    sliding_window: int,
):
    """
    Paged decode attention with block table indexing and sliding window.

    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Paged K cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Paged V cache [num_blocks, block_size, num_kv_heads, head_dim]
        o: Output tensor [batch_size, num_heads, head_dim]
        block_table: Block table [batch_size, max_num_blocks]
        seq_lens: Sequence lengths [batch_size]
        sm_scale: Softmax scale
        sliding_window: Window size (tokens), 0 or negative means no window
    """
    batch_size, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    kv_group_num = num_heads // num_kv_heads

    BLOCK_N = min(32, block_size)

    grid = (batch_size, num_heads)

    _paged_swa_decode_blocktable_kernel[grid](
        q,
        k_cache,
        v_cache,
        o,
        block_table,
        seq_lens,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        o.stride(0),
        o.stride(1),
        block_table.stride(0),
        sliding_window=sliding_window if sliding_window > 0 else 0,
        kv_group_num=kv_group_num,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
