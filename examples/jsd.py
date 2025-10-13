"""
Helion JSD (Jensen-Shannon Divergence) Example
==============================================
This example demonstrates a Helion kernel implementation of Jensen-Shannon Divergence.
JSD is commonly used in knowledge distillation for language models, where:

JSD(beta)(P || Q) = beta * KL(P || M) + (1-beta) * KL(Q || M)
where M = beta * P + (1-beta) * Q is the mixture distribution

The generalized JSD reduces to:

- Forward KL when beta = 0: KL(P || Q)

- Reverse KL when beta = 1: KL(Q || P)

- Symmetric JSD when beta = 0.5

Based on liger_kernel's JSD implementation used for knowledge distillation in language models.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# JSD Kernel
# ----------


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def jsd_forward(
    _input: Tensor,  # student predictions (input) in log-space
    target: Tensor,  # teacher targets in log-space
    shift_labels: Tensor | None = None,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    """
    Compute Jensen-Shannon Divergence loss.

    Args:
        _input: Student predictions in log-space, shape (BT, V)
        target: Teacher targets in log-space, shape (BT, V)
        shift_labels: Optional labels for masking, shape (BT,)
        beta: Coefficient for generalized JSD in [0, 1]
        ignore_index: Index to ignore in labels

    Returns:
        loss: Scalar JSD loss
        dX: Gradient of loss wrt input
    """
    BT, V = _input.shape
    assert target.shape == _input.shape, (
        f"Shape mismatch: {target.shape} != {_input.shape}"
    )
    block_size_n = hl.register_block_size(V)
    block_size_m = hl.register_block_size(BT)

    # Create output tensor for accumulating loss
    loss = torch.zeros([BT], dtype=torch.float32, device=_input.device)
    dX = torch.empty_like(loss)

    one_minus_beta = 1 - beta

    # Count non-ignored elements
    n_non_ignore = float(BT)
    if shift_labels is not None:
        n_non_ignore = float((shift_labels != ignore_index).sum().item())
        if n_non_ignore == 0:
            return torch.zeros(
                [], dtype=_input.dtype, device=_input.device
            ), torch.zeros_like(_input)

    # Process each sequence position
    for tile_bt in hl.tile(BT, block_size=block_size_m):
        # Check for label masking
        if shift_labels is not None:
            if shift_labels[tile_bt] == ignore_index:
                for tile_X in hl.tile(V):
                    dX[tile_bt, tile_X] = 0.0
                continue
        intermediate_loss = hl.zeros([tile_bt, block_size_n], dtype=torch.float32)
        intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=_input.dtype)
        for tile_v in hl.tile(V, block_size=block_size_n):
            # Load log probabilities and convert to float32
            X = _input[tile_bt, tile_v]
            Y = target[tile_bt, tile_v]

            if beta == 0.0:  # Forward KL: KL(P || Q)
                Y_max = torch.amax(Y, dim=0)
                Y_shift = Y - Y_max
                Y_prob = torch.exp(Y_shift) * torch.exp(
                    Y_max
                )  # Compensate for the shift
                intermediate_loss += Y_prob * (Y - X)
                intermediate_dX += -Y_prob
            elif beta == 1.0:  # Reverse KL: KL(Q || P)
                X_max = torch.amax(X, dim=0)
                X_shift = X - X_max
                X_prob = torch.exp(X_shift) * torch.exp(
                    X_max
                )  # Compensate for the shift
                intermediate_loss += X_prob * (X - Y)
                intermediate_dX += intermediate_loss + X_prob
            else:  # General JSD: beta*KL(P||M) + (1-beta)*KL(Q||M)
                Q = torch.exp(X)  # = exp(X)
                P = torch.exp(Y)  # = exp(Y)

                beta_P = beta * P
                one_minus_beta_Q = one_minus_beta * Q
                M = beta_P + one_minus_beta_Q
                log_M = torch.log(M)
                x_minus_log_m = X - log_M
                kl_q_m = one_minus_beta_Q * x_minus_log_m

                intermediate_loss += beta_P * (Y - log_M) + kl_q_m
                intermediate_dX += kl_q_m

        # Accumulate over vocabulary dimension
        scale = 1.0 / n_non_ignore
        loss[tile_bt] = torch.sum(intermediate_loss * scale, dim=1)
        dX[tile_bt] = torch.sum(intermediate_dX * scale, dim=1)

    # Normalize by number of non-ignored elements, run it on host to match liger_kernel
    final_loss = torch.sum(
        loss
    )  # This line raises a warning: helion.exc.TensorOperationInWrapper

    return final_loss, dX


# %%
# JSD Loss Module (matches liger_kernel structure)
# ------------------------------------------------


# %%
class HelionJSD(nn.Module):
    """
    Helion implementation of Jensen-Shannon Divergence matching liger_kernel.LigerJSD structure.

    JSD(beta)(P || Q) = beta * KL(P || M) + (1-beta) * KL(Q || M)
    where M = beta * P + (1-beta) * Q

    Args:
        beta: Coefficient beta ∈ [0,1]. When beta=0: forward KL, beta=1: reverse KL, beta=0.5: symmetric JSD
        ignore_index: Index to ignore in labels for masking
        dtype: Data type for loss computation
    """

    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        _input: Tensor,  # student predictions in log-space
        target: Tensor,  # teacher targets in log-space
        shift_labels: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass computing JSD loss.

        Args:
            _input: Student predictions in log-space, shape (BT, V)
            target: Teacher targets in log-space, shape (BT, V)
            shift_labels: Optional labels for masking, shape (BT,)
        Returns:
            Scalar JSD loss
        """
        if shift_labels is not None:
            assert shift_labels.shape == (_input.shape[0],), (
                f"the shape of shift_labels must be (BT,). Got: {shift_labels.shape}"
            )
            shift_labels = shift_labels.contiguous()
        loss, dX = jsd_forward(
            _input, target, shift_labels, self.beta, self.ignore_index
        )
        return loss.to(self.dtype)


class TorchJSDBaseline(nn.Module):
    """PyTorch baseline JSD implementation matching tritonbench."""

    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self, log_q: Tensor, log_p: Tensor, label: Tensor | None = None
    ) -> Tensor:
        # Convert to float for computation
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))

        # Mixture distribution
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)

        # JSD loss
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        if label is not None:
            loss = torch.where(label != self.ignore_index, loss, 0.0)
            n_non_ignore = (label != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                loss = torch.tensor(0.0, device=log_q.device, dtype=torch.float)
            else:
                loss = (loss / n_non_ignore).sum()
        else:
            loss = (loss / log_q.shape[0]).sum()

        return loss.to(self.dtype)


# %%
# Verification Function
# ---------------------


# %%
def check_jsd_kernel(
    B: int,
    T: int,
    V: int,
    beta: float = 0.5,
    ignore_index: int = -100,
    use_labels: bool = False,
) -> None:
    """
    Verify the JSD kernel implementation against PyTorch's baseline.

    Args:
        B: Batch size (B)
        T: Sequence length (T)
        V: Vocabulary size (V)
        beta: JSD coefficient
        ignore_index: Index to ignore in labels
        use_labels: Whether to test with label masking
    """
    # Create test tensors
    log_q = torch.randn(B * T, V, requires_grad=True, device=DEVICE).log_softmax(dim=-1)
    log_p = torch.randn(B * T, V, device=DEVICE).log_softmax(dim=-1)

    shift_labels = None
    if use_labels:
        shift_labels = torch.randint(0, V, (B,), device=DEVICE)
        # Randomly set some to ignore_index
        shift_labels[torch.rand(B, device=DEVICE) < 0.1] = -100

    # Test forward pass only (no gradients for now)
    helion_jsd = HelionJSD(beta=beta, ignore_index=ignore_index)
    torch_jsd = TorchJSDBaseline(beta=beta, ignore_index=ignore_index)

    def helion_wrapper(
        log_q: Tensor, log_p: Tensor, shift_labels: Tensor | None = None
    ) -> Tensor:
        return helion_jsd(log_q, log_p, shift_labels)

    def baseline_wrapper(
        log_q: Tensor, log_p: Tensor, shift_labels: Tensor | None = None
    ) -> Tensor:
        return torch_jsd(log_q, log_p, shift_labels)

    run_example(helion_wrapper, baseline_wrapper, (log_q, log_p, shift_labels))


# %%
# Tritonbench Integration
# -----------------------


# %%
def jsd_tritonbench(tb_op: object, log_q: Tensor, log_p: Tensor) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        log_q: Student predictions in log-space
        log_p: Teacher targets in log-space

    Returns:
        Callable: A callable that runs the JSD kernel
    """

    baseline_model = tb_op.baseline_op  # pyright: ignore[reportAttributeAccessIssue]

    helion_jsd = HelionJSD(
        beta=baseline_model.beta,
        ignore_index=baseline_model.ignore_index,
        dtype=baseline_model.dtype,
    )

    return lambda: helion_jsd(log_q, log_p)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs JSD kernel verification.
    Tests various configurations including different beta values and label masking.
    """
    print("Testing JSD kernel...")
    B = 4
    T = 2048
    beta = 0.5
    ignore_index = -100
    use_labels = False

    for V in [2**i for i in range(16, 18)]:
        print(
            f"Testing JSD: B={B}, T={T}, V={V}, beta={beta}, ignore_index={ignore_index}, labels={use_labels}"
        )
        check_jsd_kernel(B, T, V, beta, ignore_index, use_labels)
        print("✓ JSD passed")


# %%
if __name__ == "__main__":
    main()
