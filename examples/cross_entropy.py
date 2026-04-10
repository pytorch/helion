"""
Cross Entropy Loss Example
==========================

This example demonstrates how to implement a cross entropy loss function using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Cross Entropy Kernel
# --------------------


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy(
    logits: torch.Tensor,  # [N, V] input logits
    labels: torch.Tensor,  # [N] target labels
) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and target labels.

    Implements the cross entropy loss function commonly used in classification tasks.
    The function computes the log softmax of the logits and then calculates the negative
    log likelihood of the true labels.

    Args:
        logits: Input logits tensor of shape [N, V] where N is batch size and V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices

    Returns:
        A scalar tensor containing the mean cross entropy loss
    """
    n, v = logits.shape
    losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)

    for tile_n in hl.tile(n):
        # Get data for this tile
        labels_tile = labels[tile_n]  # [tile_size]

        logits_at_target = hl.zeros([tile_n], dtype=logits.dtype)
        max_logits_acc = hl.full([tile_n], float("-inf"), dtype=logits.dtype)

        # First pass: find max and target logits
        for v_chunk in hl.tile(v):
            chunk_logits = logits[tile_n, v_chunk]

            # Extract target using a chunked mask
            mask = (v_chunk.index[None, :] == labels_tile[:, None]).to(logits.dtype)
            logits_at_target += torch.sum(chunk_logits * mask, dim=-1)

            # Update max
            max_logits_acc = torch.maximum(
                max_logits_acc, torch.amax(chunk_logits, dim=-1)
            )

        # Second pass: sum exp
        sum_exp_acc = hl.zeros([tile_n], dtype=logits.dtype)
        for v_chunk in hl.tile(v):
            chunk_logits = logits[tile_n, v_chunk]
            shifted = chunk_logits - max_logits_acc[:, None]
            sum_exp_acc += torch.sum(torch.exp(shifted), dim=-1)

        log_sum_exp = max_logits_acc + torch.log(sum_exp_acc)

        # Cross entropy loss: log_sum_exp - logit_at_target
        losses[tile_n] = log_sum_exp - logits_at_target

    return losses.mean()


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the cross entropy kernel verification.
    """
    batch_size, seq_len, vocab_size = 8, 2048, 131072
    n = batch_size * seq_len
    logits = torch.randn(n, vocab_size, device=DEVICE, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=torch.int32)

    def baseline_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, labels.long())

    run_example(
        cross_entropy,
        baseline_ce,
        (logits, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
