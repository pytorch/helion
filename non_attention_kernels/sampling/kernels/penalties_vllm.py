# SPDX-License-Identifier: Apache-2.0
"""
Standalone Triton kernel for repetition penalty in sampling.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    input_ids_ptr,
    input_ids_stride,
    penalty_ptr,
    seq_len,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply repetition penalty to logits in-place."""
    row_idx = tl.program_id(0)

    # Load penalty for this row
    penalty = tl.load(penalty_ptr + row_idx)

    # Process each token in the input sequence
    for i in range(seq_len):
        token_id = tl.load(input_ids_ptr + row_idx * input_ids_stride + i)

        # Skip invalid tokens (e.g., padding with -1)
        if token_id >= 0 and token_id < vocab_size:
            # Load the logit for this token
            logit_offset = row_idx * logits_stride + token_id
            logit = tl.load(logits_ptr + logit_offset)

            # Apply repetition penalty
            # If logit > 0: logit = logit / penalty
            # If logit < 0: logit = logit * penalty
            new_logit = tl.where(logit > 0, logit / penalty, logit * penalty)

            # Store back
            tl.store(logits_ptr + logit_offset, new_logit)


def apply_penalties(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    repetition_penalty: torch.Tensor,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        input_ids: Tensor of shape [batch_size, seq_len] with token IDs
        repetition_penalty: Tensor of shape [batch_size] with penalty values (>1.0 to penalize)

    Returns:
        Penalized logits (modified in-place and returned)
    """
    batch_size, vocab_size = logits.shape
    _, seq_len = input_ids.shape
    BLOCK_SIZE = 256

    _penalties_kernel[(batch_size,)](
        logits,
        logits.stride(0),
        input_ids,
        input_ids.stride(0),
        repetition_penalty,
        seq_len,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return logits
