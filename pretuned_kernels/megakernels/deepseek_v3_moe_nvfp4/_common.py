"""Shared device-side building blocks for the DeepSeek NVFP4 kernels."""

from __future__ import annotations

import torch

import helion.language as hl

FP4_MAX = 6.0
MMA_N = 16


def fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E2M1 with FlashInfer's tie convention."""
    magnitude = torch.abs(value)
    code = (magnitude > 0.25).to(torch.int32)
    code += (magnitude >= 0.75).to(torch.int32)
    code += (magnitude > 1.25).to(torch.int32)
    code += (magnitude >= 1.75).to(torch.int32)
    code += (magnitude > 2.5).to(torch.int32)
    code += (magnitude >= 3.5).to(torch.int32)
    code += (magnitude > 5.0).to(torch.int32)
    return code | ((value < 0).to(torch.int32) << 3)


def first_mma_column(value: torch.Tensor) -> torch.Tensor:
    """Select the real token column from the padded native-MMA N tile."""
    column = hl.arange(MMA_N)
    return torch.sum(value * (column[None, :] == 0).to(torch.float32), dim=-1)


def stable_argmax_id(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> torch.Tensor:
    """Select the lowest index among equal maxima."""
    maximum = torch.amax(values, dim=-1, keepdim=True)
    return torch.amin(
        torch.where(
            values == maximum,
            indices,
            torch.full_like(indices, sentinel),
        ),
        dim=-1,
    )


def take_stable_argmax(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one stable maximum and mask it from the next selection."""
    selected = stable_argmax_id(values, indices, sentinel)
    remaining = torch.where(
        indices == selected[:, None],
        torch.full_like(values, float("-inf")),
        values,
    )
    return selected, remaining


def take_stable_top8(
    values: torch.Tensor,
    indices: torch.Tensor,
    sentinel: int,
) -> tuple[torch.Tensor, ...]:
    """Select eight stable maxima while keeping every value register-local."""
    id_0, values = take_stable_argmax(values, indices, sentinel)
    id_1, values = take_stable_argmax(values, indices, sentinel)
    id_2, values = take_stable_argmax(values, indices, sentinel)
    id_3, values = take_stable_argmax(values, indices, sentinel)
    id_4, values = take_stable_argmax(values, indices, sentinel)
    id_5, values = take_stable_argmax(values, indices, sentinel)
    id_6, values = take_stable_argmax(values, indices, sentinel)
    id_7, _values = take_stable_argmax(values, indices, sentinel)
    return id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_7
