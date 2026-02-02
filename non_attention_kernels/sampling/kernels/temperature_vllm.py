# SPDX-License-Identifier: Apache-2.0
"""
Standalone Triton kernel for temperature scaling in sampling.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply temperature scaling to logits in-place."""
    row_idx = tl.program_id(0)

    # Load temperature for this row
    temperature = tl.load(temperature_ptr + row_idx)

    # Process logits in blocks
    for block_start in range(0, vocab_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < vocab_size

        # Load logits
        logits_offset = row_idx * logits_stride + offsets
        logits = tl.load(logits_ptr + logits_offset, mask=mask)

        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Store back
        tl.store(logits_ptr + logits_offset, scaled_logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        temperature: Tensor of shape [batch_size] with temperature values

    Returns:
        Scaled logits (modified in-place and returned)
    """
    batch_size, vocab_size = logits.shape
    BLOCK_SIZE = 1024

    _temperature_kernel[(batch_size,)](
        logits,
        logits.stride(0),
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return logits
