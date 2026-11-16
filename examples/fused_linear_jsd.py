"""
Fused Linear JSD Example
========================

This example demonstrates how to implement a memory-efficient fused linear + JSD
kernel using Helion. The fusion bounds the materialized [chunk, vocab] logits
tensor (one chunk when the whole [batch, vocab] logits fit the byte budget) by:
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

# Chunking exists to bound the materialized [chunk, vocab] fp32 logits (student,
# teacher and grad_logits chunks are live at once), not to make chunks tiny:
# every chunk costs two GEMMs, a kernel launch and two more GEMMs.
LOGITS_CHUNK_BYTES = 64 * 2**20

# fast_math's approximate exp and division flush fp32 subnormal probabilities to
# zero, so the kernels clamp the mixture to fp32 tiny before taking its log
# (log 0 = -inf and p * (log p - log m) would be 0 * inf = NaN).
FP32_TINY = torch.finfo(torch.float32).tiny

# %%
# Helion JSD Kernel
# -----------------
# This kernel computes JSD loss and gradient w.r.t. logits for a single chunk.
# The gradient is computed during the forward pass to avoid storing full logits.


# %%
# fast_math: approximate exp, log and division; see FP32_TINY for the clamp
# on the mixture before its log.
@helion.kernel(fast_math=True)
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
        log_m = torch.log(torch.clamp(m, min=FP32_TINY))

        # JSD = (1-beta) * KL(student || m) + beta * KL(teacher || m)
        # KL(p || q) = sum(p * (log(p) - log(q)))
        student_kl = (student_prob * (student_log_prob - log_m)).sum(dim=-1)
        teacher_kl = (teacher_prob * (teacher_log_prob - log_m)).sum(dim=-1)
        jsd_loss = (1 - beta) * student_kl + beta * teacher_kl

        loss[tile] = jsd_loss

        # Gradient of the row's JSD w.r.t. the student logits, through the
        # softmax: d(JSD)/d(student_logits) = (1-beta)/T * p * (log(p/m) - KL(p||m))
        # (the mean over the batch is applied in the autograd backward).
        grad_tile = (
            ((1 - beta) / temperature)
            * student_prob
            * ((student_log_prob - log_m) - student_kl[:, None])
        )
        grad_student_logits[tile, :] = grad_tile

    return loss, grad_student_logits


# %%
# Backwards-compatible kernel for unit tests
# ------------------------------------------
# This kernel computes JSD loss from logits (without gradients).
# For memory-efficient fused linear + JSD, use fused_linear_jsd_fwd instead.


# %%
# fast_math: approximate exp, log and division; see FP32_TINY for the clamp
# on the mixture before its log.
@helion.kernel(fast_math=True, ignore_warnings=[helion.exc.TensorOperationInWrapper])
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
        log_m = torch.log(torch.clamp(m, min=FP32_TINY))

        # JSD = (1-beta) * KL(student || m) + beta * KL(teacher || m)
        student_kl = (student_prob * (student_log_prob - log_m)).sum(dim=-1)
        teacher_kl = (teacher_prob * (teacher_log_prob - log_m)).sum(dim=-1)
        jsd_loss = (1 - beta) * student_kl + beta * teacher_kl

        loss[tile] = jsd_loss

    return loss.mean()


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

        Instead of materializing [batch, vocab] logits beyond the byte budget, we:
        1. Process in chunks along the batch dimension
        2. Compute logits per-chunk: chunk_logits = chunk_input @ weight.T
        3. Compute JSD loss and grad_logits per-chunk
        4. Convert grad_logits to grad_input: grad_input = grad_logits @ weight

        This way we only ever have [chunk_size, vocab] logits in memory.
        """
        batch_size = student_input.shape[0]
        vocab_size = student_weight.shape[0]

        # Chunk rows so one [chunk, vocab] fp32 logits tensor fits the byte
        # budget.  Liger's vocab/hidden ratio rule made 8-16 chunks at the
        # benchmark shapes (64,128,1024), (128,512,4096), (256,1024,16384),
        # whose whole logits tensors are only 0.25-16 MiB, so the pipeline
        # launched that many small GEMMs plus a per-chunk add and ran 4-8x
        # slower than torch.compile of the unchunked reference.  Under the
        # budget those shapes are one chunk; (1024, 4096, 128256) still chunks
        # (about 130 rows per chunk).
        chunk_size = min(batch_size, max(1, LOGITS_CHUNK_BYTES // (vocab_size * 4)))
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        # Gradient buffers, written (not accumulated into) by the first chunk:
        # every row of grad_student_input belongs to exactly one chunk and the
        # weight gradient starts from the first chunk's GEMM, so neither needs
        # a zero fill.  Contiguous so the row slices are plain GEMM outputs.
        grad_student_input = student_input.new_empty(
            student_input.shape, dtype=torch.float
        )
        grad_student_weight = student_weight.new_empty(
            student_weight.shape, dtype=torch.float
        )
        loss_chunks: list[Tensor] = []

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

            loss_chunks.append(loss_chunk)

            # Convert grad_logits [chunk, vocab] to grad_input [chunk, hidden],
            # grad_input = grad_logits @ weight, written straight into its rows
            torch.mm(
                grad_logits_chunk,
                student_weight,
                out=grad_student_input[start_idx:end_idx],
            )

            # Weight gradient grad_logits.T @ input: the first chunk's GEMM
            # writes the buffer, later chunks accumulate in place so the add
            # rides the GEMM.
            if chunk_id == 0:
                torch.mm(
                    grad_logits_chunk.T,
                    student_input_chunk.float(),
                    out=grad_student_weight,
                )
            else:
                grad_student_weight.addmm_(
                    grad_logits_chunk.T, student_input_chunk.float()
                )

        # Mean of the per-row losses over the batch: one reduction kernel
        total_loss = (
            loss_chunks[0] if num_chunks == 1 else torch.cat(loss_chunks)
        ).mean()

        # Save for backward; the gradients are of the per-row loss sum, the
        # backward applies the mean's 1 / batch_size.
        ctx.save_for_backward(grad_student_input, grad_student_weight)
        ctx.batch_size = batch_size

        return total_loss

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: Any,  # noqa: ANN401
    ) -> tuple[Tensor, Tensor, None, None, None, None, None]:
        """
        Backward pass: rescale the precomputed gradients by grad_output and
        the mean's 1 / batch_size.
        """
        (grad_output,) = grad_outputs
        grad_student_input, grad_student_weight = ctx.saved_tensors

        scale = grad_output / ctx.batch_size
        grad_student_input = scale * grad_student_input
        grad_student_weight = scale * grad_student_weight

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
