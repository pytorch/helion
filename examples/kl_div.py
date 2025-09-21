"""
Helion KL Divergence Example
============================
This example demonstrates a Helion kernel implementation of Kullback-Leibler Divergence.
KL divergence is commonly used in deep learning for comparing probability distributions:

KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

When the input is in log-space (as common with log-softmax outputs):
KL(P || Q) = sum_i P(i) * (log(P(i)) - log(Q(i)))

The loss supports different reduction modes:
- 'none': No reduction, returns per-example losses
- 'sum': Sum all losses
- 'mean': Average over all elements
- 'batchmean': Average over batch dimension

Based on liger_kernel's KL divergence implementation used in language models.
"""

# %%
# Imports
# -------
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

import helion
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# KL Divergence Kernel
# --------------------
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def kl_div_forward(
    y_pred: Tensor,  # input predictions in log-space, shape (BT, V)
    y_true: Tensor,  # target values, shape (BT, V)
    log_target: bool = False,
    reduction: str = "batchmean",
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute KL Divergence loss.

    Args:
        y_pred: Input predictions in log-space, shape (BT, V)
        y_true: Target values (probabilities or log-probabilities), shape (BT, V)
        log_target: If True, y_true is in log-space; if False, y_true is probabilities
        reduction: Reduction mode ('none', 'sum', 'mean', 'batchmean')
        eps: Small value to avoid numerical issues

    Returns:
        loss: KL divergence loss
    """
    BT, V = y_pred.shape
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: {y_true.shape} != {y_pred.shape}"
    )

    # Initialize loss buffer in float32 to match Liger kernel behavior
    if reduction == "none":
        loss = torch.zeros((BT, V), dtype=torch.float32, device=y_pred.device)
    else:
        loss = torch.zeros((BT,), dtype=torch.float32, device=y_pred.device)

    # Call register_block_size to know block_size_n outside of the reduction loop.
    block_size_n = hl.register_block_size(V)

    BT_SIZE = helion.cdiv(BT, BT)  # Process all at once for simplicity
    for tile_bt in hl.tile(BT, block_size=BT_SIZE):
        loss_sum = hl.zeros([tile_bt], dtype=torch.float32)

        for tile_v in hl.tile(V, block_size=block_size_n):
            y_pred_val = y_pred[tile_bt, tile_v]
            y_true_val = y_true[tile_bt, tile_v]
            y_pred_f32 = y_pred_val.to(torch.float32)
            y_true_f32 = y_true_val.to(torch.float32)

            # Compute per-token KL contribution for the selected target space
            loss_tile = torch.zeros_like(y_pred_f32)
            if log_target:
                loss_tile = torch.exp(y_true_f32) * (y_true_f32 - y_pred_f32)
            else:
                loss_tile = y_true_f32 * (
                    torch.log(torch.clamp(y_true_f32, min=eps)) - y_pred_f32
                )

            if reduction == "none":
                loss[tile_bt, tile_v] = loss_tile
            else:
                # Sum over vocabulary dimension in float32
                loss_sum += loss_tile.sum(dim=-1)

        if reduction != "none":
            loss[tile_bt] = loss_sum

    # Apply final reduction
    if reduction == "batchmean":
        final_loss = torch.sum(loss) / BT
    elif reduction == "sum":
        final_loss = torch.sum(loss, dim=0)
    elif reduction == "mean":
        final_loss = torch.sum(loss) / (BT * V)
    else:  # reduction == "none"
        final_loss = loss

    return final_loss


# %%
# KL Divergence Loss Module
# -------------------------
class HelionKLDivLoss(nn.Module):
    """
    Helion implementation of KL Divergence Loss matching PyTorch's KLDivLoss.

    KL(P || Q) computes the divergence between target distribution P and input Q.

    Args:
        reduction: Reduction mode ('none', 'sum', 'mean', 'batchmean')
        log_target: If True, target is in log-space; if False, target is probabilities
        eps: Small value for numerical stability
    """

    def __init__(
        self,
        reduction: str = "batchmean",
        log_target: bool = False,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.eps = eps

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        """
        Forward pass computing KL divergence loss.

        Args:
            input_tensor: Input predictions in log-space, shape (BT, V)
            target_tensor: Target values (probabilities or log-probabilities), shape (BT, V)

        Returns:
            KL divergence loss
        """
        return kl_div_forward(
            input_tensor, target_tensor, self.log_target, self.reduction, self.eps
        )


# %%
# Verification Function
# ---------------------
def check_kl_div_kernel(
    B: int,
    T: int,
    V: int,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
) -> None:
    """
    Verify the KL divergence kernel implementation against PyTorch's baseline.

    Args:
        B: Batch size
        T: Sequence length
        V: Vocabulary size
        reduction: Reduction mode
        log_target: Whether target is in log-space
        eps: Small value for numerical stability
    """
    # Create test tensors following tritonbench pattern
    input_tensor = torch.randn(B * T, V, requires_grad=True, device="cuda").log_softmax(
        dim=-1
    )

    if log_target:
        target_tensor = torch.randn(B * T, V, device="cuda").log_softmax(dim=-1)
    else:
        target_tensor = torch.randn(B * T, V, device="cuda").softmax(dim=-1)

    # Test forward pass
    helion_kl = HelionKLDivLoss(reduction=reduction, log_target=log_target, eps=eps)
    torch_kl_div = torch.nn.KLDivLoss(reduction=reduction, log_target=log_target).to(
        "cuda"
    )

    def helion_wrapper(input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return helion_kl(input_tensor, target_tensor)

    def baseline_wrapper(input_tensor: Tensor, target_tensor: Tensor) -> Tensor:
        return torch_kl_div(input_tensor, target_tensor)

    run_example(helion_wrapper, baseline_wrapper, (input_tensor, target_tensor))


# %%
# Tritonbench Integration
# -----------------------
def kl_div_tritonbench(
    tb_op: object, input_tensor: Tensor, target_tensor: Tensor
) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        tb_op: Tritonbench operator object
        input_tensor: Input predictions in log-space
        target_tensor: Target values

    Returns:
        Callable: A callable that runs the KL divergence kernel
    """
    helion_kl = HelionKLDivLoss(
        reduction="batchmean",
        log_target=False,  # tritonbench uses probabilities, not log-probabilities
        eps=1e-10,
    )

    return lambda: helion_kl(input_tensor, target_tensor)


# %%
# Main Function
# -------------
def main() -> None:
    """
    Main entry point that runs KL divergence kernel verification.
    Tests various configurations matching tritonbench settings.
    """
    print("Testing KL divergence kernel...")
    B = 8
    T = 512
    reduction = "batchmean"
    log_target = False
    eps = 1e-10

    # Test with vocabulary sizes from tritonbench (2^12 to 2^17)
    for V in [2**i for i in range(12, 18)]:
        print(
            f"Testing KL Div: B={B}, T={T}, V={V}, reduction={reduction}, log_target={log_target}"
        )
        check_kl_div_kernel(B, T, V, reduction, log_target, eps)
        print("✓ KL Div passed")


# %%
if __name__ == "__main__":
    main()
