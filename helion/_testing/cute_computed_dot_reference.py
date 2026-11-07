"""Small two-dot compositions requiring completed producer coordinates."""

from __future__ import annotations

import torch

import helion.language as hl


def computed_dot_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    computed_rhs: hl.constexpr,
) -> torch.Tensor:
    output = torch.empty((a.size(0), c.size(1)), device=a.device, dtype=torch.float32)
    for row in hl.tile(a.size(0)):
        intermediate = hl.dot(a[row, :], b[:, :]).to(a.dtype)
        if computed_rhs:
            value = hl.dot(c[:, :].T, intermediate.T).T
        else:
            value = hl.dot(intermediate, c[:, :])
        output[row, :] = value
    return output


def pointwise_operand_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=torch.float32)
    for row in hl.tile(a.size(0)):
        output[row, :] = hl.dot(a[row, :] * 2, b[:, :])
    return output
