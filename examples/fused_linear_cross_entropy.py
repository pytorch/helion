"""
Fused Linear Cross Entropy Example
==================================

This example demonstrates how to implement a linear layer fused with cross entropy loss. It can
optionally emit a z-loss term as well.

This implementation is different from Liger's in that it does not use chunking / materialize logit
chunks in memory. There is a single fused fwd kernel and a single fused bwd kernel. The fwd kernel
uses an online softmax to compute the log-sum-exp across the vocabulary dimension while tracking
the target logits for each token. The bwd kernel reuses the log-sum-exp values computed in the fwd
kernel and recomputes the logits in an online manner.

This implementation is more memory efficient than Liger's and can handle larger vocabularies and
batch sizes. It does not precompute gradients in the fwd pass and so for forward-only applications
it is much faster than Liger's implementation (which does precompute gradients in the fwd pass).
Because it recomputes the logits in the bwd pass, it can be slower in some fwd-bwd scenarios.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable
from typing import Literal

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Helion Kernel
# -------------


# %%
@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper], static_shapes=False
)
def fused_linear_cross_entropy_fwd_kernel(
    inputs: torch.Tensor,  # [B, D]
    weight: torch.Tensor,  # [D, V]
    labels: torch.Tensor,  # [B]
    ignore_index: hl.constexpr,
    reduction: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    b = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    lse = torch.zeros([b], dtype=torch.float32, device=inputs.device)
    ce_loss = torch.zeros([], dtype=torch.float32, device=inputs.device)
    z_squared = torch.zeros([], dtype=torch.float32, device=inputs.device)
    if reduction == "mean":
        n_valid = torch.zeros([], dtype=torch.float32, device=inputs.device)
    else:
        n_valid = None

    for tile_b in hl.tile(b):
        labels_b = labels[tile_b]
        max_logits_b = hl.full([tile_b], float("-inf"), dtype=torch.float32)
        sum_exp_b = hl.zeros([tile_b], dtype=torch.float32)
        target_logits_b = hl.zeros([tile_b], dtype=torch.float32)
        for tile_v in hl.tile(v):
            logits_bv = hl.zeros([tile_b, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_dv = hl.load(weight, index=[tile_d, tile_v])
                logits_bv = torch.addmm(logits_bv, inputs[tile_b, tile_d], weight_dv)

            new_max = torch.maximum(max_logits_b, logits_bv.amax(dim=-1))
            scale = torch.exp(max_logits_b - new_max)
            sum_exp_b = sum_exp_b * scale + (
                torch.exp(logits_bv - new_max.unsqueeze(1)).sum(dim=-1)
            )
            max_logits_b = new_max

            # labels_bt may contain `ignore_index` values, so we need to filter them out
            is_target_in_tile = (labels_b >= tile_v.begin) & (labels_b < tile_v.end)
            local_vocab_idx = labels_b - tile_v.begin
            safe_local_vocab_idx = torch.where(is_target_in_tile, local_vocab_idx, 0)
            gathered_target_logits = (
                hl.inline_triton(  # TODO: replace with torch.gather when that works
                    "tl.sum(tl.gather({0}, {1}.to(tl.int32)[:, None], axis=1), axis=1)",
                    args=(logits_bv, safe_local_vocab_idx),
                    output_like=safe_local_vocab_idx.to(torch.float32),
                )
            )
            target_logits_b = (
                target_logits_b
                + gathered_target_logits * is_target_in_tile.to(torch.float32)
            )

        lse_b = max_logits_b + torch.log(sum_exp_b)
        ce_losses_b = lse_b - target_logits_b
        z_squared_b = lse_b.pow(2)
        lse[tile_b] = lse_b

        is_valid = (labels_b != ignore_index).to(torch.float32)
        if reduction == "sum":
            masked_ce = (ce_losses_b * is_valid).sum()
            masked_z_sq = (z_squared_b * is_valid).sum()
            hl.atomic_add(ce_loss, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
        elif reduction == "mean":
            masked_ce = (ce_losses_b * is_valid).sum()
            masked_z_sq = (z_squared_b * is_valid).sum()
            hl.atomic_add(ce_loss, [], masked_ce)
            hl.atomic_add(z_squared, [], masked_z_sq)
            hl.atomic_add(n_valid, [], is_valid.sum())
        else:
            raise NotImplementedError(
                f"Forward pass for reduction='{reduction}' not supported"
            )

    return ce_loss, z_squared, lse, n_valid


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper], static_shapes=False
)
def fused_linear_cross_entropy_bwd_kernel(
    inputs: torch.Tensor,  # [B, D]
    weight: torch.Tensor,  # [D, V]
    labels: torch.Tensor,  # [B]
    lse: torch.Tensor,  # [B]
    n_valid: torch.Tensor,
    grad_ce_loss_scalar: torch.Tensor,
    grad_z_loss_scalar: torch.Tensor,
    z_loss_multiplier: float,
    ignore_index: hl.constexpr,
    reduction: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = inputs.shape[0]
    d = hl.specialize(inputs.shape[1])
    v = hl.specialize(weight.shape[1])

    grad_weight = torch.zeros([d, v], dtype=torch.float32, device=inputs.device)
    grad_input = torch.zeros([b, d], dtype=torch.float32, device=inputs.device)
    for tile_b in hl.tile(b):
        labels_b = labels[tile_b]
        lse_b = lse[tile_b].to(torch.float32)
        is_valid = (labels_b != ignore_index).to(torch.float32)

        if reduction == "sum":
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32)
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32)
            grad_ce_per_token = is_valid * grad_ce_scalar
            grad_z_per_token = (
                is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_b
            )
        elif reduction == "mean":
            n_valid_scalar = n_valid[()].to(torch.float32)
            grad_ce_scalar = grad_ce_loss_scalar[()].to(torch.float32) / n_valid_scalar
            grad_z_scalar = grad_z_loss_scalar[()].to(torch.float32) / n_valid_scalar
            grad_ce_per_token = is_valid * grad_ce_scalar
            grad_z_per_token = (
                is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse_b
            )
        else:
            raise NotImplementedError(
                f"Backward pass for reduction='{reduction}' not supported"
            )

        for tile_v in hl.tile(v):
            logits = hl.zeros([tile_b, tile_v], dtype=torch.float32)
            for tile_d in hl.tile(d):
                weight_dv = hl.load(
                    weight, index=[tile_d, tile_v], eviction_policy="evict_last"
                )
                logits = torch.addmm(logits, inputs[tile_b, tile_d], weight_dv)

            softmax = torch.exp(logits - lse_b.unsqueeze(1))
            local_vocab_idx = labels_b - tile_v.begin
            is_target_in_tile = (labels_b >= tile_v.begin) & (labels_b < tile_v.end)
            cols = hl.arange(tile_v.block_size)
            is_target = (cols[None, :] == local_vocab_idx[:, None]) & is_target_in_tile[
                :, None
            ]
            grad_logits = softmax - is_target.to(softmax.dtype)

            grad_logits = grad_logits * grad_ce_per_token.unsqueeze(1)
            grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)
            grad_logits = grad_logits.to(inputs.dtype)

            for tile_d in hl.tile(d):
                weight_dv = hl.load(weight, index=[tile_d, tile_v])
                grad_input[tile_b, tile_d] = torch.addmm(
                    grad_input[tile_b, tile_d], grad_logits, weight_dv.T
                )
            for tile_d in hl.tile(d):
                update_gw = hl.dot(inputs[tile_b, tile_d].T, grad_logits)
                hl.atomic_add(grad_weight, [tile_d, tile_v], update_gw)

    return grad_input.to(inputs.dtype), grad_weight.to(weight.dtype)


# %%
# Autograd Function and convenience wrapper
# -----------------------------------------


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        reduction: Literal["sum", "mean"] = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 0.0,
    ) -> tuple[torch.Tensor | float, torch.Tensor | float | None]:
        ce_loss, z_squared, lse, n_valid = fused_linear_cross_entropy_fwd_kernel(
            inputs, weight, target, ignore_index, reduction
        )
        if reduction == "mean":
            ce_loss = ce_loss / n_valid
            if compute_z_loss:
                z_squared = z_squared / n_valid

        if compute_z_loss:
            z_loss = z_loss_multiplier * z_squared
        else:
            z_loss = None

        ctx.save_for_backward(inputs, weight, target, lse, n_valid)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.z_loss_multiplier = z_loss_multiplier
        return ce_loss, z_loss

    @staticmethod
    def backward(ctx, grad_ce_loss: torch.Tensor, grad_z_loss: torch.Tensor | None):
        inputs, weight, target, lse, n_valid = ctx.saved_tensors
        if grad_z_loss is None:
            grad_z_loss = torch.zeros([], dtype=lse.dtype, device=lse.device)

        grad_input, grad_weight = fused_linear_cross_entropy_bwd_kernel(
            inputs,
            weight,
            target,
            lse,
            n_valid,
            grad_ce_loss,
            grad_z_loss,
            ctx.z_loss_multiplier,
            ctx.ignore_index,
            ctx.reduction,
        )

        return (
            grad_input,  # input
            grad_weight,  # weight
            None,  # target
            None,  # ignore_index
            None,  # reduction
            None,  # compute_z_loss
            None,  # z_loss_multiplier
        )


def fused_linear_cross_entropy(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """
    Compute fused linear projection and cross-entropy loss with optional z-loss regularization.

    This function fuses the final linear layer (logits computation) with the cross-entropy
    loss calculation, avoiding materialization of the full logits tensor in memory.

    Args:
        inputs: Input tensor of shape [BT, D] where BT is the number of tokens and D is
            the hidden dimension.
        weight: Weight matrix of shape [V, D] where V is the vocabulary size.
        target: Target labels of shape [BT] containing class indices in [0, V).
        ignore_index: Target value to ignore when computing the loss. Tokens with this
            target value will not contribute to the loss or gradient. Default: -100.
        reduction: Specifies the reduction to apply to the output. One of:
            - "mean": the sum of the output will be divided by the number of valid tokens
            - "sum": the output will be summed
            Default: "mean".
        z_loss_multiplier: Coefficient for the auxiliary z-loss term, which penalizes
            large log-partition function values (log-sum-exp of logits). Set to 0.0 to disable.
            Default: 0.0.

    Returns:
        The combined cross-entropy loss plus z-loss (scaled by z_loss_multiplier).
    """
    ce_loss, z_loss = FusedLinearCrossEntropyFunction.apply(
        inputs,
        weight.T,
        target,
        ignore_index,
        reduction,
        z_loss_multiplier > 0.0,  # compute_z_loss
        z_loss_multiplier,
    )
    if z_loss is not None:
        return ce_loss + z_loss
    return ce_loss


# %%
# Benchmark Entry Point Function
# ------------------------------


# %%
def fused_linear_cross_entropy_tritonbench(
    tb_op: object, inputs: torch.Tensor, weight: torch.Tensor, target: torch.Tensor
) -> Callable[[], torch.Tensor]:
    # pyrefly: ignore [missing-attribute]
    baseline_model = tb_op.baseline_model
    ignore_index = baseline_model.ce_loss.ignore_index
    reduction = baseline_model.ce_loss.reduction

    return (
        lambda: fused_linear_cross_entropy(
            inputs, weight, target, ignore_index, reduction
        ).to(inputs.dtype)
    )  # tritonbench requires the output to be in the same dtype as the input (doesnt make sense for this kernel)


# %%
# Reference Implementation (materializes logits in memory)
# --------------------------------------------------------


# %%
@torch.compile
def linear_cross_entropy_pytorch(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    z_loss_multiplier: float = 0.0,
) -> torch.Tensor:
    """Reference implementation for a linear cross-entropy operation"""
    logits = (inputs @ weight.T).float()
    ce_loss = torch.nn.functional.cross_entropy(
        logits, target, ignore_index=ignore_index, reduction=reduction
    )
    z_loss = None
    compute_z_loss = z_loss_multiplier > 0.0
    if compute_z_loss:
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = log_z.pow(2)
        mask = target != ignore_index
        z_loss = z_loss * mask.to(z_loss.dtype)
        if reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        elif reduction == "sum":
            z_loss = z_loss.sum()
        else:
            raise NotImplementedError(f"Reduction='{reduction}' not supported")
        z_loss = z_loss_multiplier * z_loss

    if z_loss is not None:
        return ce_loss + z_loss
    return ce_loss


@torch.no_grad()
def linear_cross_entropy_fwd_pytorch(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reference implementation for forward pass of fused linear cross-entropy."""
    # Test passes weight: [V, D], transpose to [D, V] to match kernel expectation
    logits = (inputs @ weight).float()
    lse = torch.logsumexp(logits, dim=-1)  # [B]

    # Mask invalid tokens
    is_valid = (target != ignore_index).to(torch.float32)

    # Compute target logits (set to 0 for ignored tokens to match kernel behavior)
    # The kernel doesn't accumulate target_logits for ignore_index values, so they stay 0
    target_logits = torch.zeros_like(lse)  # Initialize to 0
    valid_mask = is_valid.bool()
    if valid_mask.any():
        target_logits[valid_mask] = (
            logits[valid_mask].gather(1, target[valid_mask].unsqueeze(1)).squeeze(1)
        )

    # Compute CE loss per token: lse - target_logits
    ce_losses = lse - target_logits  # [B]

    # Mask invalid tokens (multiply by is_valid)
    masked_ce_losses = ce_losses * is_valid

    # Compute z_squared per token
    z_squared_per_token = lse.pow(2) * is_valid  # [B]

    # Aggregate based on reduction
    if reduction == "sum":
        ce_loss = masked_ce_losses.sum()
        z_squared = z_squared_per_token.sum()
        n_valid = None
    elif reduction == "mean":
        ce_loss = masked_ce_losses.sum()
        z_squared = z_squared_per_token.sum()
        n_valid = is_valid.sum()
    else:
        raise NotImplementedError(f"Reduction='{reduction}' not supported")

    return ce_loss, z_squared, lse, n_valid


@torch.no_grad()
def linear_cross_entropy_bwd_pytorch(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    n_valid: torch.Tensor,
    grad_ce_loss_scalar: torch.Tensor,
    grad_z_loss_scalar: torch.Tensor,
    z_loss_multiplier: float,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for backward pass of fused linear cross-entropy."""
    logits = (inputs @ weight).float()
    is_valid = (labels != ignore_index).to(torch.float32)

    if reduction == "sum":
        grad_ce_scalar = grad_ce_loss_scalar.item()
        grad_z_scalar = grad_z_loss_scalar.item()
        grad_ce_per_token = is_valid * grad_ce_scalar
        grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse
    elif reduction == "mean":
        n_valid_scalar = n_valid.item()
        grad_ce_scalar = grad_ce_loss_scalar.item() / n_valid_scalar
        grad_z_scalar = grad_z_loss_scalar.item() / n_valid_scalar
        grad_ce_per_token = is_valid * grad_ce_scalar
        grad_z_per_token = is_valid * grad_z_scalar * z_loss_multiplier * 2.0 * lse
    else:
        raise NotImplementedError(
            f"Backward pass for reduction='{reduction}' not supported"
        )

    softmax = torch.softmax(logits, dim=-1)

    one_hot = torch.zeros_like(softmax)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    one_hot = one_hot * is_valid.unsqueeze(1)

    grad_logits = softmax - one_hot
    grad_logits = grad_logits * grad_ce_per_token.unsqueeze(1)
    grad_logits = grad_logits + softmax * grad_z_per_token.unsqueeze(1)
    grad_logits = grad_logits.to(inputs.dtype)

    grad_input = grad_logits @ weight.T
    grad_weight = inputs.T @ grad_logits

    return grad_input.to(inputs.dtype), grad_weight.to(weight.dtype)


# %%
# Verification Function
# ---------------------


# %%
def check(bt: int, d: int, v: int) -> None:
    inputs = torch.randn(
        [bt, d], device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        [v, d], device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randint(0, v, (bt,), device=DEVICE, dtype=torch.long)
    ignore_index = -100
    reduction = "mean"
    z_loss_multiplier = 0.0
    run_example(
        fused_linear_cross_entropy,
        linear_cross_entropy_pytorch,
        (inputs, weight, target, ignore_index, reduction, z_loss_multiplier),
        bwd=True,
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    check(bt=4096, d=4096, v=128256)


if __name__ == "__main__":
    main()
