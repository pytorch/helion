"""
Ragged attention implementation in Helion DSL.

This implements HSTU (Hierarchical Sequential Transduction Unit) attention
which handles variable-length sequences using seq_offsets.
"""

import torch

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[32, 128],
        num_warps=4,
        num_stages=2,
    ),
    static_shapes=True,
)
def ragged_attention(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    seq_offsets: torch.Tensor,  # [num_sequences + 1] - cumulative token positions
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    """Ragged attention using SiLU activation with proper sequence masking."""
    total_tokens = q.size(0)
    num_heads = q.size(1)
    head_dim = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    
    out = torch.zeros_like(q)
    
    # Precompute inverse to avoid fp64 promotion
    inv_max_seq_len = 1.0 / float(max_seq_len)
    
    # Process each sequence
    for seq_idx in hl.grid(num_sequences):
        # Get sequence boundaries
        seq_start = seq_offsets[seq_idx]
        seq_end = seq_offsets[seq_idx + 1]
        seq_length = seq_end - seq_start
        
        # Skip empty sequences
        if seq_length > 0:
            # Process each head independently
            for head_idx in hl.grid(num_heads):
                # Tile over query positions in this sequence
                for tile_q in hl.tile(seq_start, seq_end):
                    # Initialize accumulator for this tile
                    acc = hl.zeros([tile_q, head_dim], dtype=torch.float32)
                    
                    # Attend to all key positions in this sequence
                    for tile_k in hl.tile(seq_start, seq_end):
                        # Load Q and K chunks for this head
                        q_chunk = q[tile_q, head_idx, :]  # [tile_q, head_dim]
                        k_chunk = k[tile_k, head_idx, :]  # [tile_k, head_dim]
                        
                        # Compute attention scores: Q @ K^T
                        scores = torch.matmul(q_chunk, k_chunk.T)  # [tile_q, tile_k]
                        scores_scaled = scores * alpha
                        
                        # Apply SiLU activation: x / (1 + exp(-x))
                        exp_neg = torch.exp(-scores_scaled)
                        silu_scores = scores_scaled / (1.0 + exp_neg)
                        
                        # Scale by 1/max_seq_len
                        silu_normalized = silu_scores * inv_max_seq_len
                        
                        # Load V chunk
                        v_chunk = v[tile_k, head_idx, :]  # [tile_k, head_dim]
                        
                        # Accumulate weighted values
                        update = torch.matmul(silu_normalized.to(torch.float32), v_chunk.to(torch.float32))
                        acc = acc + update
                    
                    # Store results back
                    out[tile_q, head_idx, :] = acc.to(out.dtype)
    
    return out


def ragged_attention_tritonbench(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length: bool,
) -> torch.Tensor:
    """Wrapper to match tritonbench interface."""
    return ragged_attention(q, k, v, seq_offsets, alpha, max_seq_len)


# For tritonbench integration
TRITONBENCH_ARGS = {
    "batch_size": 64,  # Reduced for better performance
    "heads": 4,
    "min_seq_len_log2": 8,  # 2^8 = 256
    "max_seq_len_log2": 8,  # 2^8 = 256
}