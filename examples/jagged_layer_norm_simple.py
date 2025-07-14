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
    
    # Process each batch in tiles - simpler approach
    for b_idx in range(B):
        start = x_offsets[b_idx]
        end = x_offsets[b_idx + 1]
        seq_len = end - start
        
        if seq_len > 0:
            # Get batch data
            batch_data = x_data[start:end, :]
            
            # Convert to float32 for computation
            batch_float = batch_data.to(torch.float32)
            
            # Compute mean
            batch_sum = batch_float.sum()
            batch_mean = batch_sum / (seq_len * M)
            
            # Compute variance
            diff = batch_float - batch_mean
            var_sum = (diff * diff).sum()
            batch_var = var_sum / (seq_len * M)
            
            # Compute reciprocal std
            batch_rstd = torch.rsqrt(batch_var + eps)
            
            # Normalize
            normalized = (batch_float - batch_mean) * batch_rstd
            
            # Store result
            out[start:end, :] = normalized.to(x_data.dtype)
    
    return out


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