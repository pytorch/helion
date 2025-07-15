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
def jagged_softmax_kernel_simplified(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified version of jagged softmax that processes one feature at a time.
    """
    B = x_offsets.size(0) - 1
    M = x_data.size(1)
    
    # Pre-allocate output tensor
    out = torch.zeros_like(x_data)
    
    # Process each batch independently
    for b_idx in hl.grid(B):
        # Get sequence boundaries
        start_idx = x_offsets[b_idx]
        end_idx = x_offsets[b_idx + 1]
        seq_len = end_idx - start_idx
        
        if seq_len > 0:
            # Process each feature independently
            for m_idx in hl.grid(M):
                # Extract the sequence for this batch and feature
                seq_slice = x_data[start_idx:end_idx, m_idx]
                
                # Compute softmax: exp(x - max) / sum(exp(x - max))
                max_val = seq_slice.max()
                exp_vals = torch.exp(seq_slice - max_val)
                sum_exp = exp_vals.sum()
                
                # Store results
                out[start_idx:end_idx, m_idx] = exp_vals / sum_exp
    
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


def jagged_softmax_tritonbench_simplified(
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
    
    return jagged_softmax_kernel_simplified(x_values, offsets)


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
        lambda x, o: jagged_softmax_kernel_simplified(x, o),
        lambda x, o: reference_jagged_softmax_kernel_pytorch(x, o),
        (x_values, x_offsets),
    )


if __name__ == "__main__":
    main()