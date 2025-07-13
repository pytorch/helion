"""Fused linear cross entropy implementation for Helion.

This implementation uses Liger's chunking strategy to reduce memory usage
while staying within Helion's constraints.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl
from helion.utils import get_gpu_memory_info

# TritonBench configuration - adjust based on available GPU memory
if get_gpu_memory_info()[0] < 16.0:
    # Low memory configuration for GPUs with less than 16GB
    TRITONBENCH_ARGS = {"hidden_size": 2048, "vocab_size": 32000}


# Simple matmul kernel for the linear layer
@helion.kernel(static_shapes=True, dot_precision="ieee")
def linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, h = input.shape
    v, h2 = weight.shape
    assert h == h2, f"Hidden size mismatch: {h} != {h2}"
    
    logits = torch.empty([n, v], dtype=torch.float32, device=input.device)
    
    for tile_n, tile_v in hl.tile([n, v]):
        acc = hl.zeros([tile_n, tile_v], dtype=torch.float32)
        for tile_h in hl.tile(h):
            acc = torch.addmm(acc, input[tile_n, tile_h], weight[tile_v, tile_h].T)
        logits[tile_n, tile_v] = acc
    
    return logits


# Cross entropy loss kernel (based on existing cross_entropy.py)
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)

    # Pre-compute base indices
    base_indices = torch.arange(n, device=logits.device) * v
    logits_flat = logits.view(-1)

    for tile_n in hl.tile(n):
        labels_tile = labels[tile_n]
        base_indices_tile = base_indices[tile_n]
        
        # Get logits at target indices
        flat_indices = base_indices_tile + labels_tile
        logits_at_target = hl.load(logits_flat, [flat_indices])
        
        # Load the full rows for this tile
        logits_rows = logits[tile_n, :]
        
        # Compute log-sum-exp
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        
        # Cross entropy loss
        losses[tile_n] = log_sum_exp - logits_at_target

    return losses.mean()


def calculate_chunk_size(batch_size: int, hidden_size: int, vocab_size: int) -> int:
    """Calculate optimal chunk size following Liger's approach."""
    # Following Liger's logic for chunk size calculation
    inc_factor = (vocab_size + hidden_size - 1) // hidden_size
    chunk_size = max(1, batch_size // inc_factor)
    
    # Make chunk_size a power of 2 for better performance
    if chunk_size > 0:
        chunk_size = 2 ** (chunk_size.bit_length() - 1)
    else:
        chunk_size = 1
    
    # Ensure chunk_size doesn't exceed batch_size
    chunk_size = min(chunk_size, batch_size)
    
    # Cap at a reasonable maximum to avoid too small chunks
    chunk_size = min(chunk_size, 256)
    
    return chunk_size


# Fused version that uses chunking to reduce memory
def fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Fused linear + cross entropy using Liger's chunking strategy."""
    batch_size, hidden_size = input.shape
    vocab_size = weight.shape[0]
    
    # Calculate optimal chunk size
    chunk_size = calculate_chunk_size(batch_size, hidden_size, vocab_size)
    
    # If chunk size equals batch size, no chunking needed
    if chunk_size >= batch_size:
        logits = linear(input, weight)
        return cross_entropy_loss(logits, labels)
    
    # Process in chunks to reduce memory usage
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    total_loss = 0.0
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, batch_size)
        actual_chunk_size = end_idx - start_idx
        
        # Extract chunk
        input_chunk = input[start_idx:end_idx]
        labels_chunk = labels[start_idx:end_idx]
        
        # Compute logits for this chunk (only chunk_size x vocab_size memory)
        logits_chunk = linear(input_chunk, weight)
        
        # Compute loss for this chunk
        chunk_loss = cross_entropy_loss(logits_chunk, labels_chunk)
        
        # Accumulate weighted by chunk size
        total_loss += chunk_loss * actual_chunk_size
    
    # Return average loss
    return total_loss / batch_size


def fused_linear_cross_entropy_pytorch(
    input: torch.Tensor,
    weight: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """PyTorch reference implementation for fused linear cross entropy."""
    # Compute logits
    logits = torch.matmul(input, weight.T)
    # Compute cross entropy
    return torch.nn.functional.cross_entropy(logits, labels)


def main() -> None:
    """Run fused linear cross entropy benchmark with different input sizes."""
    # Test with moderate size
    n, h, v = 128, 512, 1000
    input = torch.randn(n, h, device="cuda", dtype=torch.float32)
    weight = torch.randn(v, h, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)
    
    run_example(
        fused_linear_cross_entropy,
        fused_linear_cross_entropy_pytorch,
        (input, weight, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    main()