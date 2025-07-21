"""Fused linear cross entropy implementation for Helion.

This implementation uses Liger's chunking strategy to reduce memory usage.
"""

from __future__ import annotations

import os

import torch

import helion
from helion._testing import run_example
import helion.language as hl

# TritonBench configuration
if os.environ.get("HELION_DEV_LOW_VRAM", "0") == "1":
    # Low memory configuration
    TRITONBENCH_ARGS = {"hidden_size": 2048, "vocab_size": 32000}

# Maximum chunk size (similar to Liger's MAX_FUSED_SIZE)
MAX_FUSED_SIZE = 65536 // 2


@helion.kernel(static_shapes=True)
def cross_entropy_kernel(
    logits_chunk: torch.Tensor,  # [chunk_size, vocab_size]
    target_chunk: torch.Tensor,  # [chunk_size]
    loss_chunk: torch.Tensor,  # [chunk_size]
    chunk_size: int,
    vocab_size: int,
    n_total_samples: int,  # Total number of samples for mean reduction
) -> None:
    # Grid over samples - each program handles one sample
    for program_id in hl.grid(chunk_size):
        target_idx = target_chunk[program_id].unsqueeze(0)

        # Online softmax: first pass - find max and sum
        m = hl.full([], float("-inf"))  # max value
        d = hl.full([], 0.0)  # sum of exp

        # Store original value at target
        ori_logit_y = logits_chunk[program_id, target_idx]

        # Process in blocks like Liger
        for vocab_tile in hl.tile(vocab_size):
            # Create block offsets (like tl.arange in Triton)
            block_offsets = vocab_tile.index

            # Masked load of block
            mask = block_offsets < vocab_size
            logits_block = torch.where(
                mask, logits_chunk[program_id, block_offsets], float("-inf")
            )

            # Find block max
            block_max = torch.max(logits_block)

            # Online softmax update
            m_new = torch.maximum(m, block_max)
            d = d * torch.exp(m - m_new) + torch.sum(torch.exp(logits_block - m_new))
            m = m_new

        # Compute log-sum-exp
        lse = m + torch.log(d)
        loss = lse - ori_logit_y
        # Apply mean reduction inside the kernel
        loss_chunk[program_id] = (loss / n_total_samples).squeeze(0)

        # Second pass: compute gradients with block processing
        for vocab_tile in hl.tile(vocab_size):
            block_offsets = vocab_tile.index
            mask = block_offsets < vocab_size

            # Load block
            logits_block = torch.where(
                mask, logits_chunk[program_id, block_offsets], 0.0
            )

            # Compute softmax
            softmax_block = torch.exp(logits_block - m) / d

            # Special handling for target
            is_target_block = block_offsets == target_idx
            grad_block = torch.where(
                is_target_block, softmax_block - 1.0, softmax_block
            )

            # Apply mean reduction to gradients
            grad_block = grad_block / n_total_samples

            # Masked store using torch.where pattern
            # First, load existing values for positions that will be masked out
            existing_values = logits_chunk[program_id, block_offsets]

            # Apply mask to the gradient block
            logits_chunk[program_id, block_offsets] = torch.where(
                mask, grad_block, existing_values
            )


def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Forward pass with chunking strategy similar to Liger."""
    device = _input.device
    BT, H = _input.shape
    V = weight.shape[0]

    # Calculate chunk size to limit memory usage
    inc_factor = (V + H - 1) // H
    chunk_size = min(MAX_FUSED_SIZE, (BT + inc_factor - 1) // inc_factor)
    chunk_size = min(chunk_size, BT)
    num_chunks = (BT + chunk_size - 1) // chunk_size

    # Initialize gradients and loss
    grad_input = torch.zeros_like(_input)
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # Process in chunks
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        actual_chunk_size = end_idx - start_idx

        # Get chunk of input and target
        input_chunk = _input[start_idx:end_idx]  # [chunk_size, H]
        target_chunk = target[start_idx:end_idx]  # [chunk_size]

        # Compute logits for this chunk
        logits_chunk = input_chunk @ weight.t()  # [chunk_size, V]
        if bias is not None:
            logits_chunk = logits_chunk + bias

        # Ensure contiguous for kernel
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Get loss slice
        loss_chunk = loss_1d[start_idx:end_idx]

        # Call kernel - this modifies logits_chunk in-place to contain gradients
        cross_entropy_kernel(
            logits_chunk,
            target_chunk,
            loss_chunk,
            actual_chunk_size,
            V,
            BT,  # Pass total number of samples for mean reduction
        )

        # Now logits_chunk contains gradients
        # Compute input gradient: grad_input = grad_logits @ weight
        grad_input[start_idx:end_idx] = logits_chunk.detach() @ weight.detach()

        # Accumulate weight gradients if needed
        if grad_weight is not None:
            # grad_weight += grad_logits.T @ input
            # Detach tensors to avoid autograd issues with in-place operations
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.detach().t(),
                mat2=input_chunk.detach(),
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        if grad_bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.detach().sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Return total loss
    loss = loss_1d.sum()

    return loss, grad_input, grad_weight, grad_bias


# User-facing function
def fused_linear_cross_entropy(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused linear + cross entropy."""
    loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
        input_tensor, weight, labels, bias
    )

    # For this example, we just return the loss
    # In a real implementation with autograd, we'd save gradients for backward
    return loss


def fused_linear_cross_entropy_pytorch(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference implementation for fused linear cross entropy."""
    # Compute logits
    logits = torch.matmul(input_tensor, weight.T)
    if bias is not None:
        logits = logits + bias
    # Compute cross entropy
    return torch.nn.functional.cross_entropy(logits, labels)


def main() -> None:
    n, h, v = 128, 512, 1000
    torch.manual_seed(42)
    input_tensor = torch.randn(n, h, device="cuda", dtype=torch.float32)
    weight = torch.randn(v, h, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)

    run_example(
        fused_linear_cross_entropy,
        fused_linear_cross_entropy_pytorch,
        (input_tensor, weight, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    main()
