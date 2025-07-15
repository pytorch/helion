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
else:
    # Higher memory configuration
    TRITONBENCH_ARGS = {"B": 64, "M": 16, "seqlen": 128}


@helion.kernel()
def jagged_softmax_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized jagged softmax that reduces the number of passes.
    """
    B = x_offsets.size(0) - 1
    M = x_data.size(1)
    
    # Pre-allocate output tensor
    out = torch.zeros_like(x_data)
    
    # Flatten tensors for easier indexing
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
        
        # Process all features together using tiles
        for feat_idx in hl.tile(M):
            # Combined pass: find max and compute exp sum
            max_vals = hl.full([tile_b, feat_idx], float("-inf"), dtype=torch.float32)
            
            # First pass: find maximum values
            for seq_idx in hl.tile(0, max_seq_len):
                # Mask for valid positions in each sequence
                valid_pos = (seq_idx.index[None, :] < seq_len[:, None]) & valid_batch[:, None]
                
                # Compute flattened indices
                base_indices = start_idx[:, None] + seq_idx.index[None, :]
                flat_indices = base_indices[:, :, None] * M + feat_idx.index[None, None, :]
                
                # Load values with masking
                vals = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=valid_pos[:, :, None],
                ).to(torch.float32)
                
                # Update max values - mask out invalid positions
                masked_vals = torch.where(valid_pos[:, :, None], vals, float("-inf"))
                local_max = masked_vals.amax(dim=1)
                max_vals = torch.maximum(max_vals, local_max)
            
            # Second pass: compute exp and sum
            exp_sum = hl.zeros([tile_b, feat_idx], dtype=torch.float32)
            
            for seq_idx in hl.tile(0, max_seq_len):
                valid_pos = (seq_idx.index[None, :] < seq_len[:, None]) & valid_batch[:, None]
                
                # Compute flattened indices
                base_indices = start_idx[:, None] + seq_idx.index[None, :]
                flat_indices = base_indices[:, :, None] * M + feat_idx.index[None, None, :]
                
                # Load values
                vals = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=valid_pos[:, :, None],
                ).to(torch.float32)
                
                # Compute exp(x - max) and accumulate sum
                exp_vals = torch.exp(vals - max_vals[:, None, :])
                masked_exp = torch.where(valid_pos[:, :, None], exp_vals, 0.0)
                exp_sum = exp_sum + masked_exp.sum(dim=1)
            
            # Third pass: normalize and store
            for seq_idx in hl.tile(0, max_seq_len):
                valid_pos = (seq_idx.index[None, :] < seq_len[:, None]) & valid_batch[:, None]
                
                # Compute flattened indices
                base_indices = start_idx[:, None] + seq_idx.index[None, :]
                flat_indices = base_indices[:, :, None] * M + feat_idx.index[None, None, :]
                
                # Load values
                vals = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=valid_pos[:, :, None],
                ).to(torch.float32)
                
                # Compute softmax
                exp_vals = torch.exp(vals - max_vals[:, None, :])
                softmax_vals = torch.where(
                    exp_sum[:, None, :] > 0,
                    exp_vals / exp_sum[:, None, :],
                    0.0
                )
                
                # Store results
                hl.store(
                    out_flat,
                    [flat_indices],
                    softmax_vals.to(x_data.dtype),
                    extra_mask=valid_pos[:, :, None],
                )
    
    return out


def reference_jagged_softmax_kernel_pytorch(
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation for jagged softmax."""
    B = x_offsets.numel() - 1
    out = torch.zeros_like(x_values)
    
    for i in range(B):
        start = int(x_offsets[i])
        end = int(x_offsets[i + 1])
        if end > start:
            # Get the batch's values - shape (seq_len, M)
            batch_values = x_values[start:end, :]
            # Perform softmax along dimension 0 (sequence dimension)
            softmax_values = torch.nn.functional.softmax(batch_values, dim=0)
            out[start:end, :] = softmax_values
    
    return out


def jagged_softmax_tritonbench(
    x: torch.Tensor, B: int, M: int, seqlen: int, sparsity: float
) -> torch.Tensor:
    """
    Wrapper for tritonbench that matches the expected interface.
    """
    x_values = x.values()
    x_offsets = x.offsets()
    
    # x_offsets might be a tuple of offsets for multiple dimensions
    # For softmax, we use the first dimension's offsets
    if isinstance(x_offsets, tuple):
        offsets = x_offsets[0]
    else:
        offsets = x_offsets
    
    # Force compilation before benchmark
    if not hasattr(jagged_softmax_kernel, '_compiled'):
        print(f"[DEBUG] First call - compiling kernel for B={B}, M={M}, seqlen={seqlen}")
        jagged_softmax_kernel._compiled = True
    
    return jagged_softmax_kernel(x_values, offsets)


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
        lambda x, o: jagged_softmax_kernel(x, o),
        lambda x, o: reference_jagged_softmax_kernel_pytorch(x, o),
        (x_values, x_offsets),
    )


if __name__ == "__main__":
    main()