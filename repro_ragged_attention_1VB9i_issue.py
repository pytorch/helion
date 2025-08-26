"""
Simplified Jagged HSTU Attention Forward Example
===============================================

This example demonstrates a simplified version of jagged HSTU attention using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import helion
import helion.language as hl
from helion._testing import run_example

import torch

def generate_small_inputs():
    """Generate small inputs for HSTU attention for easier verification and testing"""

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    batch_size = 2
    max_seq_len = 128  # N parameter
    heads = 4
    head_dim = 128
    
    # Generate random sequence lengths between 25% and 100% of max_seq_len
    min_seq_len = max_seq_len // 2
    seq_lengths = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    seq_offsets = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), 
                            torch.cumsum(seq_lengths, dim=0)])
    total_seq_len = int(seq_offsets[-1].item())

    # Generate tensors with actual ragged sequence length
    # q, k, v: [total_seq_len, heads, head_dim]
    q = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    # Parameters
    alpha = 1.0 / (head_dim**0.5)  # Scaling factor
    invalid_attn_mask_type = "lower_triangular"

    # Optional parameters (set to None for simplicity)
    num_targets = None
    attn_scale = None
    attn_bias = None
    seq2_offsets = None

    # Integer parameters
    max_attn_len = 0
    contextual_seq_len = 0
    sort_by_length = False
    full_attn_size = 0

    return {
        "N": max_seq_len,
        "alpha": alpha,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": seq_offsets,
        "invalid_attn_mask_type": invalid_attn_mask_type,
        "num_targets": num_targets,
        "attn_scale": attn_scale,
        "attn_bias": attn_bias,
        "seq2_offsets": seq2_offsets,
        "max_attn_len": max_attn_len,
        "contextual_seq_len": contextual_seq_len,
        "sort_by_length": sort_by_length,
        "full_attn_size": full_attn_size,
    }


def reference_jagged_hstu_kernel_pytorch(inputs):
    """Simple PyTorch implementation of HSTU ragged attention using direct tensor slicing"""
    q = inputs["q"]  # [total_seq_len, heads, head_dim]
    k = inputs["k"]  # [total_seq_len, heads, head_dim]
    v = inputs["v"]  # [total_seq_len, heads, head_dim]
    seq_offsets = inputs["seq_offsets"]  # [batch_size + 1]
    alpha = inputs["alpha"]
    N = inputs["N"]  # max_seq_len

    # Initialize output
    output = torch.zeros_like(v)

    # Scale factor
    scale = 1.0 / N

    # Compute per-batch sequence lengths
    seq_lens = seq_offsets[1:] - seq_offsets[:-1]

    q_split = torch.split(q, seq_lens.tolist(), dim=0)
    k_split = torch.split(k, seq_lens.tolist(), dim=0)
    v_split = torch.split(v, seq_lens.tolist(), dim=0)

    # Process each sequence in the batch using direct tensor slicing
    for i, (q_batch, k_batch, v_batch) in enumerate(
        zip(q_split, k_split, v_split, strict=False)
    ):
        q_batch = q_batch.transpose(0, 1)  # [heads, seq_len, head_dim]
        k_batch = k_batch.permute(1, 2, 0)  # [heads, head_dim, seq_len]
        v_batch = v_batch.transpose(0, 1)  # [heads, seq_len, head_dim]

        # Compute attention scores using batch matrix multiplication
        scores = torch.bmm(q_batch, k_batch) * alpha

        # Add attention bias if provided (not used for now)

        # Apply SiLU activation
        scores = (scores / (1.0 + torch.exp(-scores))) * scale

        # Apply invalid mask
        if inputs["invalid_attn_mask_type"] == "lower_triangular":
            invalid_mask = torch.tril(
                torch.ones_like(scores, dtype=torch.bool), diagonal=0
            )
            scores = torch.where(invalid_mask, scores, torch.zeros_like(scores))

        # Compute output using batch matrix multiplication
        # bmm: [heads, seq_len, seq_len] @ [heads, seq_len, head_dim] -> [heads, seq_len, head_dim]
        output_batch = torch.bmm(scores, v_batch)

        # Transpose back and store result: [heads, seq_len, head_dim] -> [seq_len, heads, head_dim]
        output[seq_offsets[i] : seq_offsets[i + 1]] = output_batch.transpose(0, 1)

    return output


@helion.kernel(static_shapes=True)
def _helion_ragged_attention_kernel(
    q: torch.Tensor,  # [total_seq_len, num_heads, q_dim]
    k: torch.Tensor,  # [total_seq_len, num_heads, k_dim]
    v: torch.Tensor,  # [total_seq_len, num_heads, v_dim]
    seq_offsets: torch.Tensor,  # [batch_size + 1]
    alpha: float,
    invalid_mask_type: str,
    max_seq_len_tensor: torch.Tensor,
) -> torch.Tensor:  
    max_seq_len = max_seq_len_tensor.numel()
    scale = 1.0 / max_seq_len  # attn_scale is None

    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))

    out = torch.zeros_like(v)

    for tile_b, tile_h, tile_q in hl.tile([num_batches, num_heads, max_seq_len], block_size=[1, 1, None]):
        grid_b = tile_b.begin
        grid_h = tile_h.begin

        starts = seq_offsets[grid_b]  # [grid_b]
        ends = seq_offsets[grid_b + 1]  # [grid_b]
        seq_len = ends - starts  # [grid_b]

        if tile_q.begin < seq_len:
            q_blk = q[tile_q.index + starts, grid_h, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)

            if invalid_mask_type == "lower_triangular":
                low = 0
                high = tile_q.end
                
            for tile_kv in hl.tile(low, high, block_size=None):
                
                k_blk = k[tile_kv.index + starts, grid_h, :]
                v_blk = v[tile_kv.index + starts, grid_h, :]
                        
                scores = torch.nn.functional.silu(hl.dot(q_blk, k_blk.T) * alpha) * scale
                
                if invalid_mask_type == "lower_triangular":
                    mask_q = tile_q.index < seq_len
                    mask_kv = tile_kv.index < seq_len
                    scores = torch.where((tile_q.index[:, None] > tile_kv.index[None, :]) & mask_q[:, None] & mask_kv[None, :], scores, 0.0)

                acc = torch.addmm(acc, scores.to(v.dtype), v_blk)
                
            # Store result
            out[tile_q.index + starts, grid_h, :] = acc
      
    return out


def helion_ragged_attention_function(inputs):
    """
    Wrapper function for the Helion ragged attention kernel.
    """

    return _helion_ragged_attention_kernel(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        seq_offsets=inputs["seq_offsets"],
        alpha=inputs["alpha"],
        invalid_mask_type=inputs["invalid_attn_mask_type"],
        max_seq_len_tensor=torch.empty(inputs["N"], device=inputs["q"].device),
    )


def main() -> None:
    """
    Main entry point for testing the simplified jagged HSTU attention kernel.
    """
    inputs = generate_small_inputs()
    run_example(
        lambda inputs: helion_ragged_attention_function(inputs),
        lambda inputs: reference_jagged_hstu_kernel_pytorch(inputs),
        (inputs,)
    )


if __name__ == "__main__":
    main()
