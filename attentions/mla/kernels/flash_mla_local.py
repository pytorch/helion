# Local flash_mla implementation
# Provides flash_mla-compatible API using local Triton kernels

from typing import Tuple, Optional
import math
import torch

from kernelkit_utils import cdiv


class TileSchedulerMetadata:
    """Metadata for tile scheduling (placeholder for local implementation)."""

    def __init__(self, batch_size: int, num_splits: int):
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.tile_scheduler_metadata = None


def get_mla_metadata(
    batch_size: int = 128,
    num_splits: int = 8,
) -> Tuple[TileSchedulerMetadata, int]:
    """
    Get MLA metadata for kernel scheduling.

    This is a simplified version that returns placeholder metadata.
    The actual flash_mla uses this for tile scheduling optimization.

    Returns:
        (tile_scheduler_metadata, num_splits)
    """
    metadata = TileSchedulerMetadata(batch_size, num_splits)
    return metadata, num_splits


def flash_mla_with_kvcache(
    q: torch.Tensor,           # [batch, s_q, h_q, d]
    blocked_kv: torch.Tensor,  # [num_blocks, block_size, h_kv, d]
    block_table: torch.Tensor, # [batch, max_num_blocks]
    cache_seqlens: torch.Tensor,  # [batch]
    dv: int,                   # Value head dimension
    tile_scheduler_metadata: TileSchedulerMetadata,
    num_splits: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash MLA attention with KV cache.

    This implements the flash_mla.flash_mla_with_kvcache API using local kernels.

    For MLA:
    - Q has shape [batch, s_q, h_q, d] where d=576 typically (512 latent + 64 RoPE)
    - KV has shape [num_blocks, block_size, h_kv, d]
    - Output has shape [batch, s_q, h_q, dv] where dv=512 typically

    Args:
        q: Query tensor [batch, s_q, h_q, d]
        blocked_kv: Blocked KV cache [num_blocks, block_size, h_kv, d]
        block_table: Block indices per batch [batch, max_num_blocks]
        cache_seqlens: Sequence lengths [batch]
        dv: Value dimension
        tile_scheduler_metadata: Scheduling metadata (unused in local impl)
        num_splits: Number of KV splits
        causal: Whether to use causal masking

    Returns:
        (output, lse): Output tensor and log-sum-exp
    """
    batch, s_q, h_q, d = q.shape
    block_size = blocked_kv.shape[1]
    h_kv = blocked_kv.shape[2]
    device = q.device
    dtype = q.dtype

    # Allocate output tensors
    out = torch.zeros(batch, s_q, h_q, dv, dtype=dtype, device=device)
    lse = torch.zeros(batch, h_q, s_q, dtype=torch.float32, device=device)

    # Process each query position
    # For simplicity, we'll use a PyTorch implementation that matches flash_mla semantics
    # This is slower than the Triton kernel but ensures correctness

    cache_seqlens_cpu = cache_seqlens.cpu()
    scale = d ** -0.5

    for b in range(batch):
        cur_len = int(cache_seqlens_cpu[b].item())
        if cur_len == 0:
            # No KV cache - output zeros, lse=+inf (no attention)
            lse[b] = float('+inf')
            continue

        # Gather KV for this batch from blocked storage
        num_blocks = cdiv(cur_len, block_size)
        block_indices = block_table[b, :num_blocks]

        # Gather blocks and reshape to [seq_len, h_kv, d]
        kv_blocks = blocked_kv[block_indices]  # [num_blocks, block_size, h_kv, d]
        kv = kv_blocks.reshape(-1, h_kv, d)[:cur_len]  # [cur_len, h_kv, d]

        # Split K and V (in MLA, both use same tensor but different dims)
        k = kv  # [cur_len, h_kv, d]
        v = kv[..., :dv]  # [cur_len, h_kv, dv]

        # Convert to float32 for numerical stability
        q_b = q[b].float()  # [s_q, h_q, d]
        k_f = k.float()  # [cur_len, h_kv, d]
        v_f = v.float()  # [cur_len, h_kv, dv]

        # Expand KV heads for GQA/MQA
        kv_group_num = h_q // h_kv
        if kv_group_num > 1:
            k_f = k_f.unsqueeze(2).expand(-1, -1, kv_group_num, -1)  # [cur_len, h_kv, group, d]
            k_f = k_f.reshape(cur_len, h_q, d)  # [cur_len, h_q, d]
            v_f = v_f.unsqueeze(2).expand(-1, -1, kv_group_num, -1)  # [cur_len, h_kv, group, dv]
            v_f = v_f.reshape(cur_len, h_q, dv)  # [cur_len, h_q, dv]

        for sq in range(s_q):
            q_sq = q_b[sq] * scale  # [h_q, d]

            # Compute attention scores
            # attn_scores: [h_q, cur_len]
            attn_scores = torch.einsum('hd, nhd -> hn', q_sq, k_f)

            # Apply causal mask
            if causal and s_q > 1:
                # Causal: position sq can attend to positions [0, sq + (cur_len - s_q)]
                # This handles the case where s_q < cur_len
                attendable_len = cur_len - s_q + sq + 1
                if attendable_len < cur_len:
                    mask = torch.ones(cur_len, dtype=torch.bool, device=device)
                    mask[attendable_len:] = False
                    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            # Softmax
            max_scores = attn_scores.max(dim=-1, keepdim=True).values  # [h_q, 1]
            exp_scores = torch.exp(attn_scores - max_scores)  # [h_q, cur_len]
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # [h_q, 1]
            attn_weights = exp_scores / sum_exp  # [h_q, cur_len]

            # Compute output
            # out_sq: [h_q, dv]
            out_sq = torch.einsum('hn, nhd -> hd', attn_weights, v_f)

            out[b, sq] = out_sq.to(dtype)
            lse[b, :, sq] = (max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1)))

    return out, lse


def flash_mla_with_kvcache_triton(
    q: torch.Tensor,           # [batch, s_q, h_q, d]
    blocked_kv: torch.Tensor,  # [num_blocks, block_size, h_kv, d]
    block_table: torch.Tensor, # [batch, max_num_blocks]
    cache_seqlens: torch.Tensor,  # [batch]
    dv: int,                   # Value head dimension
    tile_scheduler_metadata: TileSchedulerMetadata,
    num_splits: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flash MLA attention using local Triton decode kernel.

    This version uses the triton_mla_decode kernel for better performance,
    but requires reshaping data to match the kernel's expected format.
    """
    from triton_mla_decode import decode_attention_fwd

    batch, s_q, h_q, d = q.shape
    block_size = blocked_kv.shape[1]
    h_kv = blocked_kv.shape[2]
    device = q.device
    dtype = q.dtype

    # For s_q > 1, we need to process each query position separately
    # since the decode kernel assumes s_q=1
    if s_q > 1:
        # Fall back to PyTorch implementation for prefill-like scenarios
        return flash_mla_with_kvcache(
            q, blocked_kv, block_table, cache_seqlens,
            dv, tile_scheduler_metadata, num_splits, causal
        )

    # s_q == 1: Can use Triton decode kernel
    q_decode = q.squeeze(1)  # [batch, h_q, d]

    # The Triton kernel expects:
    # - k_buffer: [total_tokens, page_size, h_kv, d]
    # - v_buffer: [total_tokens, page_size, h_kv, dv]
    # - req_to_token: [batch, max_seq_len] mapping to token indices

    # Convert blocked format to the expected format
    # blocked_kv: [num_blocks, block_size, h_kv, d]
    # We treat each block as a "page" of tokens
    num_blocks = blocked_kv.shape[0]
    page_size = block_size

    # Reshape: [num_blocks, block_size, h_kv, d] -> [num_blocks * block_size, 1, h_kv, d]
    # But the kernel expects page_size in dim 1, so we keep block_size there
    k_buffer = blocked_kv.unsqueeze(1)  # [num_blocks, 1, block_size, h_kv, d]
    k_buffer = k_buffer.reshape(num_blocks * block_size, 1, h_kv, d)

    v_buffer = blocked_kv[..., :dv].unsqueeze(1)
    v_buffer = v_buffer.reshape(num_blocks * block_size, 1, h_kv, dv)

    # Create req_to_token mapping
    # block_table[b, i] gives the block index for batch b, block i
    # We need to convert this to token indices
    max_seq_len = int(cache_seqlens.max().item())
    req_to_token = torch.zeros(batch, max_seq_len, dtype=torch.int32, device=device)

    cache_seqlens_cpu = cache_seqlens.cpu()
    for b in range(batch):
        cur_len = int(cache_seqlens_cpu[b].item())
        num_blocks_b = cdiv(cur_len, block_size)
        for i in range(num_blocks_b):
            block_idx = block_table[b, i].item()
            start_tok = i * block_size
            end_tok = min(start_tok + block_size, cur_len)
            for t in range(start_tok, end_tok):
                # Token t maps to block_idx * block_size + (t - start_tok)
                req_to_token[b, t] = block_idx * block_size + (t - start_tok)

    # Allocate outputs
    out = torch.zeros(batch, h_q, dv, dtype=dtype, device=device)
    lse = torch.zeros(batch, h_q, dtype=torch.float32, device=device)
    attn_logits = torch.zeros(batch, h_q, num_splits, dv + 1, dtype=torch.float32, device=device)

    sm_scale = d ** -0.5

    # Run Triton kernel
    decode_attention_fwd(
        q_decode, k_buffer, v_buffer, out, lse,
        req_to_token, cache_seqlens.int(), attn_logits,
        num_splits, sm_scale, page_size=1,
    )

    # Reshape output: [batch, h_q, dv] -> [batch, 1, h_q, dv]
    out = out.unsqueeze(1)
    # LSE: [batch, h_q] -> [batch, h_q, 1]
    lse = lse.unsqueeze(-1)

    return out, lse
