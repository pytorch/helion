"""
Fused Linear JSD Example
========================

This example demonstrates how to implement a JSD kernel using Helion and
fuse it with a linear layer.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Helion Kernel
# -------------


# %%
@helion.kernel()
def fused_linear_jsd_kernel(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    loss = student_logits.new_empty(student_logits.shape[0], dtype=torch.float)
    for batch in hl.tile(student_logits.shape[0]):
        student_prob = torch.log_softmax(student_logits[batch, :] / temperature, dim=-1)
        teacher_prob = torch.log_softmax(teacher_logits[batch, :] / temperature, dim=-1)
        student_prob = student_prob.to(torch.float).view(-1, student_prob.size(-1))
        teacher_prob = teacher_prob.to(torch.float).view(-1, teacher_prob.size(-1))
        m = torch.exp(student_prob) + beta * (
            torch.exp(teacher_prob) - torch.exp(student_prob)
        )
        teacher_div = torch.nn.functional.kl_div(
            torch.log(m), teacher_prob, reduction="none", log_target=True
        ).sum(dim=-1)
        student_div = torch.nn.functional.kl_div(
            torch.log(m), student_prob, reduction="none", log_target=True
        ).sum(dim=-1)
        batch_loss = student_div + beta * (teacher_div - student_div)
        loss[batch] = batch_loss
    return (loss / student_logits.shape[0]).sum()


def fused_linear_jsd_fwd(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    student_input: torch.Tensor,
    teacher_input: torch.Tensor,
) -> torch.Tensor:
    student_logits = student_input @ student_weight.T
    teacher_logits = teacher_input @ teacher_weight.T
    return fused_linear_jsd_kernel(
        beta, ignore_index, temperature, student_logits, teacher_logits
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
    baseline_op = tb_op.baseline_op  # pyright: ignore[reportAttributeAccessIssue]
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
    student_logits = student_input @ student_weight.T
    teacher_logits = teacher_input @ teacher_weight.T
    student_prob = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = torch.log_softmax(teacher_logits / temperature, dim=-1)
    student_prob = student_prob.to(torch.float).view(-1, student_prob.size(-1))
    teacher_prob = teacher_prob.to(torch.float).view(-1, teacher_prob.size(-1))
    m = torch.exp(student_prob) + beta * (
        torch.exp(teacher_prob) - torch.exp(student_prob)
    )
    teacher_div = torch.nn.functional.kl_div(
        torch.log(m), teacher_prob, reduction="none", log_target=True
    ).sum(dim=-1)
    student_div = torch.nn.functional.kl_div(
        torch.log(m), student_prob, reduction="none", log_target=True
    ).sum(dim=-1)
    loss = student_div + beta * (teacher_div - student_div)
    return (loss / student_logits.shape[0]).sum()


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
        (0.5, 1, 1.0, student_weight, teacher_weight, student_input, teacher_input),
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    check(1024, 4096, 128256)


if __name__ == "__main__":
    main()
