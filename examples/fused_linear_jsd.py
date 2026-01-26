"""
Fused Linear JSD Example
========================

This example demonstrates how to implement a memory-efficient fused linear + JSD
kernel using Helion. The fusion avoids materializing the full [batch, vocab] logits
tensor by:
1. Chunking the batch dimension
2. Computing logits per-chunk via matmul
3. Computing JSD loss and grad_logits per-chunk
4. Converting grad_logits back to grad_input (batch x hidden) via matmul
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Any
from typing import Callable

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.exc
import helion.language as hl

# %%
# Helion JSD Kernel
# -----------------
# This kernel computes JSD loss and gradient w.r.t. logits for a single chunk.
# The gradient is computed during the forward pass to avoid storing full logits.


# %%
@helion.kernel()
def jsd_kernel(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute JSD loss and gradient w.r.t. student logits for a chunk.

    Args:
        beta: Interpolation coefficient for JSD (0 = student only, 1 = teacher only)
        ignore_index: Label index to ignore (unused in label-free version)
        temperature: Temperature for softmax
        student_logits: Student model logits [chunk_size, vocab_size]
        teacher_logits: Teacher model logits [chunk_size, vocab_size]

    Returns:
        loss: Per-sample JSD loss [chunk_size]
        grad_student_logits: Gradient w.r.t. student logits [chunk_size, vocab_size]
    """
    chunk_size, vocab_size = student_logits.size()
    loss = student_logits.new_empty(chunk_size, dtype=torch.float)
    grad_student_logits = torch.empty_like(student_logits, dtype=torch.float)

    for tile in hl.tile(chunk_size):
        # Scale by temperature
        student_scaled = student_logits[tile, :] / temperature
        teacher_scaled = teacher_logits[tile, :] / temperature

        # Compute softmax probabilities
        student_prob = torch.softmax(student_scaled.to(torch.float), dim=-1)
        teacher_prob = torch.softmax(teacher_scaled.to(torch.float), dim=-1)

        # Compute log probabilities for KL divergence
        student_log_prob = torch.log_softmax(student_scaled.to(torch.float), dim=-1)
        teacher_log_prob = torch.log_softmax(teacher_scaled.to(torch.float), dim=-1)

        # Compute mixture distribution: m = (1-beta)*student + beta*teacher
        m = (1 - beta) * student_prob + beta * teacher_prob
        log_m = torch.log(m)

        # JSD = (1-beta) * KL(student || m) + beta * KL(teacher || m)
        # KL(p || q) = sum(p * (log(p) - log(q)))
        student_kl = (student_prob * (student_log_prob - log_m)).sum(dim=-1)
        teacher_kl = (teacher_prob * (teacher_log_prob - log_m)).sum(dim=-1)
        jsd_loss = (1 - beta) * student_kl + beta * teacher_kl

        loss[tile] = jsd_loss

        # Gradient of JSD w.r.t. student logits
        # d(JSD)/d(student_logits) = (1-beta)/T * (student_prob - m)
        # This comes from the chain rule through softmax
        grad_tile = ((1 - beta) / temperature) * (student_prob - m)
        grad_student_logits[tile, :] = grad_tile

    return loss, grad_student_logits


# %%
# Backwards-compatible kernel for unit tests
# ------------------------------------------
# This kernel computes JSD loss from logits (without gradients).
# For memory-efficient fused linear + JSD, use fused_linear_jsd_fwd instead.


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fused_linear_jsd_kernel(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute JSD loss from pre-computed logits.

    This kernel is kept for unit testing the JSD computation in isolation.
    For memory-efficient fused linear + JSD, use fused_linear_jsd_fwd instead.
    """
    batch_size, _vocab_size = student_logits.size()
    loss = student_logits.new_empty(batch_size, dtype=torch.float)

    for tile in hl.tile(batch_size):
        # Scale by temperature
        student_scaled = student_logits[tile, :] / temperature
        teacher_scaled = teacher_logits[tile, :] / temperature

        # Compute softmax probabilities
        student_prob = torch.softmax(student_scaled.to(torch.float), dim=-1)
        teacher_prob = torch.softmax(teacher_scaled.to(torch.float), dim=-1)

        # Compute log probabilities for KL divergence
        student_log_prob = torch.log_softmax(student_scaled.to(torch.float), dim=-1)
        teacher_log_prob = torch.log_softmax(teacher_scaled.to(torch.float), dim=-1)

        # Compute mixture distribution: m = (1-beta)*student + beta*teacher
        m = (1 - beta) * student_prob + beta * teacher_prob
        log_m = torch.log(m)

        # JSD = (1-beta) * KL(student || m) + beta * KL(teacher || m)
        student_kl = (student_prob * (student_log_prob - log_m)).sum(dim=-1)
        teacher_kl = (teacher_prob * (teacher_log_prob - log_m)).sum(dim=-1)
        jsd_loss = (1 - beta) * student_kl + beta * teacher_kl

        loss[tile] = jsd_loss

    return (loss / batch_size).sum()


# %%
# Autograd Function
# -----------------
# This implements the memory-efficient forward/backward using chunking.


# %%
class FusedLinearJSDFunction(torch.autograd.Function):
    """
    Memory-efficient fused linear + JSD using chunking.

    The forward pass computes gradients w.r.t. inputs, avoiding the need to
    store the full [batch, vocab] logits tensor for backward.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        student_input: Tensor,
        student_weight: Tensor,
        teacher_input: Tensor,
        teacher_weight: Tensor,
        beta: float,
        ignore_index: int,
        temperature: float,
    ) -> Tensor:
        """
        Forward pass with chunking to reduce memory usage.

        Instead of materializing full [batch, vocab] logits, we:
        1. Process in chunks along the batch dimension
        2. Compute logits per-chunk: chunk_logits = chunk_input @ weight.T
        3. Compute JSD loss and grad_logits per-chunk
        4. Convert grad_logits to grad_input: grad_input = grad_logits @ weight

        This way we only ever have [chunk_size, vocab] in memory, not [batch, vocab].
        """
        batch_size, hidden_dim = student_input.shape
        vocab_size = student_weight.shape[0]

        # Calculate chunk size to balance memory vs overhead
        # Larger chunks = more memory but fewer kernel launches
        # Use heuristic similar to Liger: chunk based on vocab/hidden ratio
        inc_factor = max(1, (vocab_size + hidden_dim - 1) // hidden_dim)
        chunk_size = 1
        while chunk_size * 2 <= batch_size // inc_factor:
            chunk_size *= 2
        chunk_size = max(1, min(chunk_size, batch_size))
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        # Allocate outputs
        total_loss = torch.tensor(0.0, device=student_input.device, dtype=torch.float)
        grad_student_input = torch.zeros_like(student_input, dtype=torch.float)
        grad_student_weight = torch.zeros_like(student_weight, dtype=torch.float)

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min((chunk_id + 1) * chunk_size, batch_size)

            # Get chunk of inputs
            student_input_chunk = student_input[start_idx:end_idx]
            teacher_input_chunk = teacher_input[start_idx:end_idx]

            # Compute logits for this chunk only: [chunk, vocab]
            student_logits_chunk = student_input_chunk @ student_weight.T
            teacher_logits_chunk = teacher_input_chunk @ teacher_weight.T

            # Compute JSD loss and gradient w.r.t. logits
            loss_chunk, grad_logits_chunk = jsd_kernel(
                beta,
                ignore_index,
                temperature,
                student_logits_chunk,
                teacher_logits_chunk,
            )

            # Accumulate loss
            total_loss = total_loss + loss_chunk.sum()

            # Convert grad_logits [chunk, vocab] to grad_input [chunk, hidden]
            # grad_input = grad_logits @ weight
            grad_student_input[start_idx:end_idx] = grad_logits_chunk @ student_weight

            # Accumulate weight gradient: grad_weight += grad_logits.T @ input
            grad_student_weight = (
                grad_student_weight + grad_logits_chunk.T @ student_input_chunk.float()
            )

        # Normalize loss by batch size
        total_loss = total_loss / batch_size

        # Save for backward
        ctx.save_for_backward(grad_student_input, grad_student_weight)

        return total_loss

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: Any,  # noqa: ANN401
    ) -> tuple[Tensor, Tensor, None, None, None, None, None]:
        """
        Backward pass: rescale the precomputed gradients by grad_output.
        """
        (grad_output,) = grad_outputs
        grad_student_input, grad_student_weight = ctx.saved_tensors

        # Scale gradients by upstream gradient
        grad_student_input = grad_output * grad_student_input
        grad_student_weight = grad_output * grad_student_weight

        # Return gradients for: student_input, student_weight, teacher_input,
        # teacher_weight, beta, ignore_index, temperature
        return grad_student_input, grad_student_weight, None, None, None, None, None


# %%
# Forward Function
# ----------------


# %%
def fused_linear_jsd_fwd(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    student_input: torch.Tensor,
    teacher_input: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fused linear + JSD loss.

    This is the main entry point that uses the memory-efficient autograd function.
    """
    return FusedLinearJSDFunction.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        beta,
        ignore_index,
        temperature,
    )


# %%
# Benchmark Entry Point Function
# ------------------------------


# %%
def fused_linear_jsd_fwd_tritonbench(
    tb_op: object,
    student_input: torch.Tensor,
    teacher_input: torch.Tensor,
    label: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    assert label is None
    # pyrefly: ignore [missing-attribute]
    baseline_op = tb_op.baseline_op
    beta = baseline_op.jsd.beta
    ignore_index = baseline_op.jsd.ignore_index
    temperature = baseline_op.temperature
    student_weight = baseline_op.student_lin.weight
    teacher_weight = baseline_op.teacher_lin.weight
    return lambda: fused_linear_jsd_fwd(
        beta,
        ignore_index,
        temperature,
        student_weight,
        teacher_weight,
        student_input,
        teacher_input,
    )


# %%
# Reference Implementation
# ------------------------


# %%
def fused_linear_jsd_pytorch(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    student_input: torch.Tensor,
    teacher_input: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation for verification."""
    student_logits = student_input @ student_weight.T
    teacher_logits = teacher_input @ teacher_weight.T

    # Scale by temperature
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature

    # Compute probabilities
    student_prob = torch.softmax(student_scaled.to(torch.float), dim=-1)
    teacher_prob = torch.softmax(teacher_scaled.to(torch.float), dim=-1)

    # Compute log probabilities
    student_log_prob = torch.log_softmax(student_scaled.to(torch.float), dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_scaled.to(torch.float), dim=-1)

    # Mixture distribution
    m = (1 - beta) * student_prob + beta * teacher_prob
    log_m = torch.log(m)

    # JSD = (1-beta) * KL(student || m) + beta * KL(teacher || m)
    student_kl = (student_prob * (student_log_prob - log_m)).sum(dim=-1)
    teacher_kl = (teacher_prob * (teacher_log_prob - log_m)).sum(dim=-1)
    loss = (1 - beta) * student_kl + beta * teacher_kl

    return loss.mean()


# %%
# Verification Function
# ---------------------


# %%
def check(m: int, n: int, k: int) -> None:
    student_input = torch.rand([m, n], device=DEVICE, dtype=torch.float)
    teacher_input = torch.rand([m, n], device=DEVICE, dtype=torch.float)
    student_weight = torch.rand([k, n], device=DEVICE, dtype=torch.float)
    teacher_weight = torch.rand([k, n], device=DEVICE, dtype=torch.float)
    run_example(
        fused_linear_jsd_fwd,
        fused_linear_jsd_pytorch,
        (0.5, -100, 1.0, student_weight, teacher_weight, student_input, teacher_input),
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    check(1024, 4096, 128256)


if __name__ == "__main__":
    main()
