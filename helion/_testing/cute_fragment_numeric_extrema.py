"""Numeric extrema shared with a complete, two-axis logical vote."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def fragment_numeric_extremum(
    x: torch.Tensor, y: torch.Tensor, minimum: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(x)
    vote = torch.empty((1,), device=x.device, dtype=torch.bool)
    for batch in hl.grid(1):
        rows = hl.arange(x.size(0))
        columns = hl.arange(x.size(1))
        left = x[rows[:, None], columns[None, :]]
        right = y[columns]
        if minimum:
            result = torch.fmin(left, right)
        else:
            result = torch.fmax(left, right)
        output[rows[:, None], columns[None, :]] = result
        vote[batch] = torch.any(torch.isnan(result))
    return output, vote
