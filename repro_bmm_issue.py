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
import torch
def generate_small_inputs():
    """Generate small inputs for HSTU attention for easier verification and testing"""
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    # Small parameters for easier verification
    batch_size = 2
    max_seq_len = 38  # N parameter
    heads = 4
    head_dim = 16
    total_seq_len = batch_size * max_seq_len  # Total sequence length across all batches
    # Generate small tensors with hardcoded values for easier verification
    # q, k, v: [total_seq_len, heads, head_dim]
    q = torch.full(
        (total_seq_len, heads, head_dim),
        0.1,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    k = torch.full(
        (total_seq_len, heads, head_dim),
        0.2,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    v = torch.full(
        (total_seq_len, heads, head_dim),
        0.5,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    # seq_offsets: cumulative sequence offsets for each batch
    # Shape: [batch_size + 1] where seq_offsets[i+1] - seq_offsets[i] = sequence length for batch i
    seq_offsets = torch.arange(
        0, (batch_size + 1) * max_seq_len, max_seq_len, dtype=torch.int32, device=device
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
def torch_ragged_attention_function(inputs):
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
@helion.kernel(use_default_config=True)
def _helion_ragged_attention_kernel(
    q: torch.Tensor,  # [total_seq_len, num_heads, q_dim]
    k: torch.Tensor,  # [total_seq_len, num_heads, k_dim]  (k_dim == q_dim)
    v: torch.Tensor,  # [total_seq_len, num_heads, v_dim]
    seq_offsets: torch.Tensor,  # [batch_size + 1]
    alpha: float,
    scale: float,
    invalid_mask: torch.Tensor,  # [max_seq_len, max_seq_len], True=keep, False=zero
) -> torch.Tensor:
    total_seq_len = q.size(0)
    num_heads = q.size(1)
    q_dim = q.size(2)
    v_dim = v.size(2)
    # Flatten tensors for easier indexing
    q_flat = q.view(-1)  # [total_seq_len * num_heads * q_dim]
    k_flat = k.view(-1)  # [total_seq_len * num_heads * q_dim]
    v_flat = v.view(-1)  # [total_seq_len * num_heads * v_dim]
    # Output: [total_seq_len, num_heads, v_dim]
    out = torch.zeros([total_seq_len, num_heads, v_dim], dtype=q.dtype, device=q.device)
    out_flat = out.view(-1)  # [total_seq_len * num_heads * v_dim]
    # Batch of ragged sequences
    num_seqs = seq_offsets.size(0) - 1
    # --- Tile over sequences (B) ---
    for tile_b in hl.tile(num_seqs):
        starts = seq_offsets[tile_b]  # [tile_B]
        ends = seq_offsets[tile_b.index + 1]  # [tile_B]
        lens = ends - starts  # [tile_B]
        max_L = lens.amax()
        # --- Tile over heads (H) ---
        for tile_h in hl.tile(num_heads):
            # --- Tile over sequence position for Q and V (I) ---
            for tile_i in hl.tile(0, max_L):
                i_valid_bi = tile_i.index[None, :] < lens[:, None]  # [tile_B, tile_I]
                i_abs_bi = starts[:, None] + tile_i.index[None, :]  # [tile_B, tile_I]
                # --- Tile over sequence position for K (J) ---
                for tile_j in hl.tile(0, max_L):
                    j_valid_bj = tile_j.index[None, :] < lens[:, None]  # [tile_B, tile_J]
                    j_abs_bj = starts[:, None] + tile_j.index[None, :]  # [tile_B, tile_J]
                    # Build local attention mask from provided invalid_mask
                    keep_ij = hl.load(invalid_mask, [tile_i.index, tile_j.index])  # [tile_I, tile_J]
                    keep_mask = (
                        keep_ij[None, None, :, :]
                        & i_valid_bi[:, None, :, None]
                        & j_valid_bj[:, None, None, :]
                    )  # [tile_B, tile_H, tile_I, tile_J]
                    # Initialize the scores submatrix
                    scores = hl.zeros(
                        [tile_b, tile_h, tile_i, tile_j], device=q.device, dtype=torch.float32
                    )  # [tile_B, tile_H, tile_I, tile_J]
                    # --- Tile over Q and K's dimension (D) ---
                    for tile_d in hl.tile(q_dim):
                        # Compute flat indices for Q and K
                        q_flat_indices = (
                            i_abs_bi[:, None, :, None] * (num_heads * q_dim)
                            + tile_h.index[None, :, None, None] * q_dim
                            + tile_d.index[None, None, None, :]
                        )  # [tile_B, tile_H, tile_I, tile_D]
                        k_flat_indices = (
                            j_abs_bj[:, None, None, :] * (num_heads * q_dim)
                            + tile_h.index[None, :, None, None] * q_dim
                            + tile_d.index[None, None, :, None]
                        )  # [tile_B, tile_H, tile_D, tile_J]
                        # Load Q and K blocks
                        q_blk = hl.load(
                            q_flat,
                            [q_flat_indices],
                            extra_mask=i_valid_bi[:, None, :, None],
                        )  # [tile_B, tile_H, tile_I, tile_D]
                        k_blk = hl.load(
                            k_flat,
                            [k_flat_indices],
                            extra_mask=j_valid_bj[:, None, None, :],
                        )  # [tile_B, tile_H, tile_D, tile_J]
                        # Batch matrix multiply and acculate to scroes

                        # VERSION 1: This works because torch.bmm expects 3D tensors and operates on batched matrices.
                        # By reshaping to merge batch and head dimensions, we satisfy bmm's requirements.
                        # The reshape operations preserve tile metadata through Helion's compilation.
                        # prod = torch.bmm(
                        #     q_blk.reshape(-1, tile_i.block_size, tile_d.block_size),
                        #     k_blk.reshape(-1, tile_d.block_size, tile_j.block_size),
                        # ).reshape(
                        #     tile_b.block_size, tile_h.block_size, tile_i.block_size, tile_j.block_size
                        # )
                        
                        # VERSION 1: Using bmm with explicit reshaping works reliably
                        # This approach preserves tile metadata throughout the operation
                        prod = torch.bmm(
                            q_blk.reshape(-1, tile_i.block_size, tile_d.block_size),
                            k_blk.reshape(-1, tile_d.block_size, tile_j.block_size),
                        ).reshape(
                            tile_b.block_size, tile_h.block_size, tile_i.block_size, tile_j.block_size
                        )

                        scores += prod
                    # Apply alpha scaling
                    scores *= alpha
                    # Apply SiLU activation: x * sigmoid(x)
                    scores = (scores / (1.0 + torch.exp(-scores))) * scale
                    # Apply attention mask
                    scores = torch.where(keep_mask, scores, 0)
                    # --- Tile over V's dimension (V) ---
                    for tile_v in hl.tile(v_dim):
                        # Compute flat indices for V
                        v_flat_indices = (
                            j_abs_bj[:, None, :, None] * (num_heads * v_dim)
                            + tile_h.index[None, :, None, None] * v_dim
                            + tile_v.index[None, None, None, :]
                        )  # [tile_B, tile_H, tile_J, tile_V]
                        # Load V block
                        v_blk = hl.load(
                            v_flat,
                            [v_flat_indices],
                            extra_mask=j_valid_bj[:, None, :, None],
                        )  # [tile_B, tile_H, tile_J, tile_V]
                        # comptue the contribution to the output
                        contrib = torch.bmm(
                            scores.reshape(-1, tile_i.block_size, tile_j.block_size),
                            v_blk.reshape(-1, tile_j.block_size, tile_v.block_size).to(torch.float32),
                        ).reshape(
                            tile_b.block_size, tile_h.block_size, tile_i.block_size, tile_v.block_size
                        )  # [tile_B, tile_H, tile_I, tile_V]
                        # Apply validity mask and accumulate into output
                        masked_contrib = torch.where(
                            i_valid_bi[:, None, :, None], contrib, 0
                        )
                        # Use flat indexing approach for storing results
                        out_flat_indices = (
                            i_abs_bi[:, None, :, None] * (num_heads * v_dim)
                            + tile_h.index[None, :, None, None] * v_dim
                            + tile_v.index[None, None, None, :]
                        )  # [tile_B, tile_H, tile_I, tile_V]
                        
                        hl.atomic_add(out_flat, [out_flat_indices], masked_contrib)
    return out
def helion_ragged_attention_function(inputs):
    """
    Wrapper function for the Helion ragged attention kernel.
    Takes a dictionary of inputs and calls the kernel with individual arguments.
    """
    # Extract required parameters from inputs
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    seq_offsets = inputs["seq_offsets"]
    alpha = inputs["alpha"]
    N = inputs["N"]
    # Scale factor (same as PyTorch implementation)
    scale = 1.0 / N
    # Create invalid mask based on mask type
    invalid_attn_mask_type = inputs.get("invalid_attn_mask_type", "lower_triangular")
    if invalid_attn_mask_type == "lower_triangular":
        # Create lower triangular mask (True = keep, False = zero)
        invalid_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=0)
    else:
        # Default to all ones (no masking)
        invalid_mask = torch.ones(N, N, dtype=torch.bool, device=q.device)
    # Call the actual Helion kernel
    return _helion_ragged_attention_kernel(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        alpha=alpha,
        scale=scale,
        invalid_mask=invalid_mask,
    )
# %%
# Test Function
# -----------
def test_simple_jagged_hstu_attention():
    """
    Test the simplified jagged HSTU attention implementation using small inputs from fbcode.
    """
    print("=== Simplified Jagged HSTU Attention Test ===\n")
    # Generate small inputs using the fbcode function
    inputs = generate_small_inputs()
    print("Test data shapes:")
    print(f"  q: {inputs['q'].shape}")
    print(f"  k: {inputs['k'].shape}")
    print(f"  v: {inputs['v'].shape}")
    print(f"  seq_offsets: {inputs['seq_offsets']}")
    print(f"  alpha: {inputs['alpha']}")
    print(f"  N: {inputs['N']}")
    # Helion implementation
    print("\nRunning Helion implementation...")
    helion_out = helion_ragged_attention_function(inputs)
    print("Helion output type:", helion_out.dtype, "\n")
    # PyTorch reference (from fbcode)
    print("Running PyTorch reference implementation...")
    pytorch_out = torch_ragged_attention_function(inputs)
    print("\nOutput ranges:")
    print(f"  Helion: [{helion_out.min():.4f}, {helion_out.max():.4f}]")
    print(f"  PyTorch: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}]")
    # Compare results
    max_diff = torch.max(torch.abs(helion_out - pytorch_out)).item()
    mean_diff = torch.mean(torch.abs(helion_out - pytorch_out)).item()
    print("\nComparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    # Print sample outputs for debugging
    print("\nSample outputs [:5, 0, 0]:")
    print(f"  Helion:  {helion_out[:5, 0, 0]}")
    print(f"  PyTorch: {pytorch_out[:5, 0, 0]}")
    if max_diff < 1e-2:  # More lenient threshold for bfloat16
        print("✅ Test passed!")
        return True
    print(f"❌ Test failed with max diff: {max_diff}")
    # Print more detailed comparison for debugging
    print("\nDetailed comparison (first sequence, first head, first 3 elements):")
    for i in range(min(3, helion_out.size(0))):
        print(f"  Position {i}:")
        print(f"    Helion:  {helion_out[i, 0, :3]}")
        print(f"    PyTorch: {pytorch_out[i, 0, :3]}")
        print(f"    Diff:    {torch.abs(helion_out[i, 0, :3] - pytorch_out[i, 0, :3])}")
    return False
# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point for testing the simplified jagged HSTU attention kernel.
    """
    test_simple_jagged_hstu_attention()
if __name__ == "__main__":
    main()
