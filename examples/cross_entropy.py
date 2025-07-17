from __future__ import annotations

import os

import helion
import helion.language as hl

import torch
from helion._testing import run_example

# TritonBench configuration - adjust based on HELION_DEV_LOW_VRAM environment variable
if os.environ.get("HELION_DEV_LOW_VRAM", "0") == "1":
    # Low memory configuration
    TRITONBENCH_ARGS = {"B": 4, "T": 512, "v_range": "10,15"}


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

    # Flatten logits once at the beginning
    logits_flat = logits.view(-1)

    for tile_n in hl.tile(n):
        # Get data for this tile
        labels_tile = labels[tile_n]  # [tile_size]
        base_indices_tile = tile_n.index * v  # [tile_size]

        # Compute the actual flat indices by adding the label offset
        flat_indices = base_indices_tile + labels_tile

        # Load the logits at the target indices
        logits_at_target = hl.load(logits_flat, [flat_indices])

        # Compute log_softmax for numerical stability
        # Load the full rows for this tile
        logits_rows = logits[tile_n, :]  # [tile_size, V]

        # Compute log-sum-exp
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

        # Cross entropy loss: log_sum_exp - logit_at_target
        losses[tile_n] = log_sum_exp - logits_at_target

    return losses.mean()


def main() -> None:
    """Run cross entropy benchmark with different input sizes."""
    # Test with moderate size
    n, v = 128, 1000
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)

    run_example(
        cross_entropy,
        torch.nn.functional.cross_entropy,
        (logits, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
