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
    x_values: torch.Tensor,
    x_offsets: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """
    Jagged softmax kernel optimized for tritonbench.
    Processes each (batch, feature) block in parallel.
    """
    B = x_offsets.size(0) - 1
    M = x_values.size(1)
    
    # Pre-allocate output tensor
    out = torch.zeros_like(x_values)
    
    # Process each batch and feature block in a grid
    # This creates B * M parallel tasks
    for pid in hl.grid(B * M):
        # Decompose the program ID into batch and feature indices
        bid = pid // M  
        mid = pid % M
        
        # Get sequence boundaries for this batch
        start = x_offsets[bid]
        end = x_offsets[bid + 1]
        
        # Skip if empty sequence
        if end > start:
            # Initialize max value with the first element
            max_val = x_values[start, mid]
            
            # Find maximum - start from second element
            for i in range(1, end - start):
                idx = start + i
                val = x_values[idx, mid]
                max_val = torch.maximum(max_val, val)
            
            # Compute exp sum
            exp_sum = 0.0
            for i in range(end - start):
                idx = start + i
                val = x_values[idx, mid]
                exp_sum = exp_sum + torch.exp(val - max_val)
            
            # Normalize and store
            for i in range(end - start):
                idx = start + i
                val = x_values[idx, mid]
                out[idx, mid] = torch.exp(val - max_val) / exp_sum
    
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
    
    return jagged_softmax_kernel(x_values, offsets, seqlen)


def main() -> None:
    B, M = 8, 16
    device = "cuda"
    
    # Create random sequence lengths
    lengths = torch.randint(1, 32, (B,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    total_elements = int(x_offsets[-1])
    max_seqlen = int(lengths.max())
    
    # Create random values
    x_values = torch.randn(total_elements, M, dtype=torch.float32, device=device)
    
    run_example(
        lambda x, o: jagged_softmax_kernel(x, o, max_seqlen),
        lambda x, o: reference_jagged_softmax_kernel_pytorch(x, o),
        (x_values, x_offsets),
    )


if __name__ == "__main__":
    main()