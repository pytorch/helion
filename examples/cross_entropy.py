"""Cross entropy with index computation that works within Helion's constraints."""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl
from helion.utils import get_gpu_memory_info

# TritonBench configuration - adjust based on available GPU memory
if get_gpu_memory_info()[0] < 16.0:
    # Low memory configuration for GPUs with less than 16GB
    TRITONBENCH_ARGS = {"B": 4, "T": 512, "v_range": "10,15"}


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy(
    logits: torch.Tensor,  # [N, V] input logits
    labels: torch.Tensor,  # [N] target labels
) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)

    # Pre-compute base indices: [0, V, 2V, 3V, ...]
    base_indices = torch.arange(n, device=logits.device) * v

    # Flatten logits once at the beginning
    logits_flat = logits.view(-1)

    for tile_n in hl.tile(n):
        # Get data for this tile
        labels_tile = labels[tile_n]  # [tile_size]
        base_indices_tile = base_indices[tile_n]  # [tile_size]

        # Compute the actual flat indices by adding the label offset
        # flat_index[i] = base_indices[i] + labels[i] = i*V + labels[i]
        flat_indices = base_indices_tile + labels_tile

        # Load the logits at the target indices
        logits_at_target = hl.load(logits_flat, [flat_indices])

        # Now we need to compute log_softmax for numerical stability
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


def cross_entropy_pytorch(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for cross entropy."""
    return torch.nn.functional.cross_entropy(logits, labels)


def main() -> None:
    """Run cross entropy benchmark with different input sizes."""
    # Test with moderate size
    n, v = 128, 1000
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.long)

    run_example(
        cross_entropy,
        cross_entropy_pytorch,
        (logits, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
