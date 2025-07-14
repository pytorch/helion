from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl
from helion.utils import get_gpu_memory_info

# TritonBench configuration - adjust based on available GPU memory
if get_gpu_memory_info()[0] < 16.0:
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
    
    This implementation attempts to closely follow the Triton kernel implementation,
    but encounters several fundamental limitations in Helion DSL.
    
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
    
    out = torch.zeros_like(x_data)
    
    # ========================================================================
    # ATTEMPT 1: Direct Translation of Triton Kernel
    # ========================================================================
    # The Triton kernel uses:
    # - tl.program_id(0) to get batch index -> In Helion: hl.tile(B)
    # - Scalar accumulator for mean/variance -> In Helion: hl.zeros([1])
    # - Direct range loops over seq_len -> In Helion: hl.tile() required
    # - Block tiling over M dimension -> In Helion: hl.tile(M)
    
    # Process each batch (equivalent to Triton's grid over B)
    for tile_b in hl.tile(B):
        # Get start and end offsets for this batch
        start_offset = x_offsets[tile_b]
        end_offset = x_offsets[tile_b.index + 1]
        seq_len = end_offset - start_offset
        
        # PROBLEM 1: Early exit for empty sequences
        # Triton: if seq_len == 0: return
        # Helion: Cannot use conditional returns within tiles
        # Workaround: Use masking throughout computation
        
        # PROBLEM 2: Scalar accumulators
        # Triton: mean_acc = tl.zeros([1], dtype=tl.float32)
        # Helion: hl.zeros([1]) creates a tensor, not a scalar
        # This leads to broadcasting issues in accumulation
        mean_acc = hl.zeros([tile_b], dtype=torch.float32)
        
        # ========================================================================
        # FIRST PASS: Compute Mean
        # ========================================================================
        # PROBLEM 3: Dynamic range loops
        # Triton: for seq_idx in range(seq_len):
        # Helion: Cannot use range(seq_len) where seq_len is tensor-dependent
        # Must use: hl.tile(0, seq_len.amax()) with masking
        
        # PROBLEM 4: seq_len.amax() requires shape specialization
        # When seq_len is computed from tensor operations, using it in hl.tile
        # triggers ShapeSpecializingCall error
        max_seq_len = seq_len.amax()
        
        for tile_seq in hl.tile(0, max_seq_len):
            # PROBLEM 5: Complex masking for variable-length sequences
            # Need to check if current index is within valid sequence
            seq_mask = tile_seq.index < seq_len[:, None]
            
            # Compute row index (equivalent to row_idx = start_offset + seq_idx)
            row_idx = start_offset[:, None] + tile_seq.index[None, :]
            
            # Process M dimension in blocks
            for tile_m in hl.tile(M):
                # PROBLEM 6: Flattened indexing
                # Triton: values_ptr + row_idx * M + m_offs
                # Helion: Must compute flattened indices explicitly
                # This creates complex 3D index tensors
                indices = row_idx[:, :, None] * M + tile_m.index[None, None, :]
                
                # Load values with masking
                vals = hl.load(
                    x_data.view(-1),
                    [indices],
                    extra_mask=seq_mask[:, :, None],
                ).to(torch.float32)
                
                # PROBLEM 7: Accumulation with variable scoping
                # Triton: mean_acc += tl.sum(vals, axis=0)
                # Helion: Variable updates across loop iterations cause scoping issues
                # The generated Triton code loses track of variables defined in
                # previous iterations
                mean_acc = mean_acc + vals.sum(dim=2).sum(dim=1)
        
        # Compute mean
        total_elements = (seq_len * M).to(torch.float32)
        mean = torch.where(seq_len > 0, mean_acc / total_elements, 0.0)
        
        # ========================================================================
        # SECOND PASS: Compute Variance
        # ========================================================================
        # PROBLEM 8: Multi-pass algorithms with shared state
        # The 'mean' computed in the first pass needs to be used in the second pass
        # This creates complex variable dependencies that Helion's compiler
        # doesn't handle well, leading to undefined variable errors in generated code
        
        var_acc = hl.zeros([tile_b], dtype=torch.float32)
        
        # PROBLEM 9: Repeated loop structure
        # Having multiple similar loops (for mean, variance, normalization)
        # exacerbates the variable scoping issues
        for tile_seq in hl.tile(0, max_seq_len):
            seq_mask = tile_seq.index < seq_len[:, None]
            row_idx = start_offset[:, None] + tile_seq.index[None, :]
            
            for tile_m in hl.tile(M):
                indices = row_idx[:, :, None] * M + tile_m.index[None, None, :]
                
                vals = hl.load(
                    x_data.view(-1),
                    [indices],
                    extra_mask=seq_mask[:, :, None],
                ).to(torch.float32)
                
                # Compute squared differences
                diff = vals - mean[:, None, None]
                var_acc = var_acc + (diff * diff).sum(dim=2).sum(dim=1)
        
        # Compute variance and reciprocal std
        var = torch.where(seq_len > 0, var_acc / total_elements, 1.0)
        rstd = torch.rsqrt(var + eps)
        
        # ========================================================================
        # THIRD PASS: Normalize and Store
        # ========================================================================
        # PROBLEM 10: Complex indexing for stores
        # Storing to non-contiguous memory locations with masking
        # creates additional complexity in the generated code
        
        for tile_seq in hl.tile(0, max_seq_len):
            seq_mask = tile_seq.index < seq_len[:, None]
            row_idx = start_offset[:, None] + tile_seq.index[None, :]
            
            for tile_m in hl.tile(M):
                indices = row_idx[:, :, None] * M + tile_m.index[None, None, :]
                
                vals = hl.load(
                    x_data.view(-1),
                    [indices],
                    extra_mask=seq_mask[:, :, None],
                ).to(torch.float32)
                
                # Normalize
                normalized = (vals - mean[:, None, None]) * rstd[:, None, None]
                
                # Store normalized values
                hl.store(
                    out.view(-1),
                    [indices],
                    normalized.to(x_data.dtype),
                    extra_mask=seq_mask[:, :, None],
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