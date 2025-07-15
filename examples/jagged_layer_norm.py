from __future__ import annotations

import os
import torch

import helion
from helion._testing import run_example
import helion.language as hl

# TritonBench configuration - adjust based on HELION_DEV_LOW_VRAM environment variable
if os.environ.get("HELION_DEV_LOW_VRAM", "0") == "1":
    # Low memory configuration
    TRITONBENCH_ARGS = {"B": 32, "M": 8, "seqlen": 64}


@helion.kernel()
def jagged_layer_norm_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Perform layer normalization on a jagged tensor.
    
    This implementation fixes variable scoping issues by using tile-sized
    accumulators following the pattern from jagged_mean and moe_matmul_ogs.
    
    Args
    ----
    x_data : 2-D tensor of shape (total_elements, M) holding all elements.
    x_offsets : (B + 1) tensor. Batch i is the slice x_data[x_offsets[i] : x_offsets[i+1], :].
    eps : Small value for numerical stability.
    
    Returns
    -------
    result : 2-D tensor of shape (total_elements, M) containing the normalized values.
    """
    B = x_offsets.size(0) - 1
    M = x_data.size(1)
    
    # Pre-allocate output tensor
    out = torch.zeros_like(x_data)
    
    # Flatten x_data for easier indexing (following jagged_mean pattern)
    x_flat = x_data.view(-1)
    out_flat = out.view(-1)
    
    # Process each batch using tiles
    for tile_b in hl.tile(B):
        # Get batch boundaries
        start_idx = x_offsets[tile_b]
        end_idx = x_offsets[tile_b.index + 1]
        seq_len = end_idx - start_idx
        
        # Create masks for valid sequences
        valid_batch = seq_len > 0
        
        # Get the maximum sequence length in this tile
        max_seq_len = seq_len.amax()
        
        # Initialize accumulators using tile shapes (following jagged_mean pattern)
        sum_acc = hl.zeros([tile_b], dtype=torch.float32)
        sum_sq_acc = hl.zeros([tile_b], dtype=torch.float32)
        
        # Single pass to compute both sum and sum of squares
        for seq_idx in hl.tile(0, max_seq_len):
            # Mask for valid positions in each sequence
            valid_pos = (seq_idx.index[None, :] < seq_len[:, None]) & valid_batch[:, None]
            
            for feat_idx in hl.tile(M):
                # Compute flattened indices (following jagged_mean pattern)
                base_indices = start_idx[:, None] + seq_idx.index[None, :]
                flat_indices = base_indices[:, :, None] * M + feat_idx.index[None, None, :]
                
                # Combined mask for row and feature validity
                combined_mask = valid_pos[:, :, None]
                
                # Load values with masking
                vals = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                ).to(torch.float32)
                
                # Sum across sequence and feature dimensions (one at a time)
                vals_sum = vals.sum(dim=2).sum(dim=1)  # Sum features first, then seq
                vals_sq_sum = (vals * vals).sum(dim=2).sum(dim=1)
                
                sum_acc = sum_acc + vals_sum
                sum_sq_acc = sum_sq_acc + vals_sq_sum
        
        # Compute statistics
        n_elements = (seq_len * M).to(torch.float32)
        mean = torch.where(valid_batch, sum_acc / n_elements, 0.0)
        mean_sq = torch.where(valid_batch, sum_sq_acc / n_elements, 0.0)
        variance = torch.where(valid_batch, mean_sq - mean * mean, 1.0)
        rstd = torch.rsqrt(variance + eps)
        
        # Second pass: normalize and store
        for seq_idx in hl.tile(0, max_seq_len):
            valid_pos = (seq_idx.index[None, :] < seq_len[:, None]) & valid_batch[:, None]
            
            for feat_idx in hl.tile(M):
                # Compute flattened indices
                base_indices = start_idx[:, None] + seq_idx.index[None, :]
                flat_indices = base_indices[:, :, None] * M + feat_idx.index[None, None, :]
                
                combined_mask = valid_pos[:, :, None]
                
                # Load values
                vals = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                ).to(torch.float32)
                
                # Normalize: (x - mean) * rstd
                # Broadcast mean and rstd to match vals shape
                mean_expanded = mean[:, None, None]
                rstd_expanded = rstd[:, None, None]
                normalized = (vals - mean_expanded) * rstd_expanded
                
                # Store normalized values
                hl.store(
                    out_flat,
                    [flat_indices],
                    normalized.to(x_data.dtype),
                    extra_mask=combined_mask,
                )
    
    return out

# ========================================================================
# SUMMARY OF HELION LIMITATIONS FOR JAGGED LAYER NORM:
# ========================================================================
# 1. No early exit from tiles (cannot return conditionally)
# 2. Scalar accumulators become tensors, causing broadcasting complexity
# 3. Cannot use dynamic ranges in loops (range(seq_len) not allowed)
# 4. Shape specialization errors when using tensor-dependent values in tiles
# 5. Complex masking required for variable-length sequences
# 6. Flattened indexing creates 3D index tensors (complexity explosion)
# 7. Variable scoping issues across loop iterations in generated Triton code
# 8. Multi-pass algorithms with shared state cause undefined variable errors
# 9. Repeated similar loops exacerbate scoping issues
# 10. Complex non-contiguous stores with masking add to code generation complexity
#
# The fundamental issue is that Helion's code generation doesn't properly handle:
# - Variables that persist across loop iterations
# - Complex nested loops with accumulators
# - Multi-pass algorithms over the same data
# 
# This results in generated Triton code where variables like 'v_4', 'v_15' etc.
# are referenced before being defined, causing NameError at compilation.


def reference_jagged_layer_norm_kernel_pytorch(
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PyTorch reference implementation for jagged layer norm."""
    B = x_offsets.numel() - 1
    out = torch.zeros_like(x_values)
    
    for i in range(B):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            # Get the batch's values
            batch_values = x_values[start:end, :]
            # Perform layer normalization over all elements in the batch
            normalized = torch.nn.functional.layer_norm(
                batch_values.view(-1), 
                normalized_shape=[batch_values.numel()],
                eps=eps
            ).view(batch_values.shape)
            out[start:end, :] = normalized
    
    return out


def jagged_layer_norm_tritonbench(
    x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> torch.Tensor:
    """
    Wrapper for tritonbench that matches the expected interface.
    
    Args:
        x: Nested tensor in jagged format with shape (B, *, M)
        B: Batch size
        M: Number of features
        seqlen: Maximum sequence length
        sparsity: Sparsity factor (not used)
    
    Returns:
        Normalized values tensor
    """
    x_values = x._values
    x_offsets = x._offsets  # pyright: ignore[reportAttributeAccessIssue]
    
    # x_offsets might be a tuple of offsets for multiple dimensions
    # For layer norm, we use the first dimension's offsets
    if isinstance(x_offsets, tuple):
        offsets = x_offsets[0]
    else:
        offsets = x_offsets
    
    return jagged_layer_norm_kernel(x_values, offsets, eps=1e-6)


def main() -> None:
    B, M = 8, 16
    device = "cuda"
    
    # Create random sequence lengths
    lengths = torch.randint(1, 32, (B,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    total_elements = int(x_offsets[-1])
    
    # Create random values
    x_values = torch.randn(total_elements, M, dtype=torch.float32, device=device)
    
    run_example(
        lambda x, o: jagged_layer_norm_kernel(x, o, eps=1e-6),
        lambda x, o: reference_jagged_layer_norm_kernel_pytorch(x, o, eps=1e-6),
        (x_values, x_offsets),
    )


if __name__ == "__main__":
    main()