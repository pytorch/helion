"""Ordinary branch stores and escaping lazy values for fragment lifetime tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def branch_scans(
    x: torch.Tensor, flag: int, escape: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    first_out = torch.empty_like(x)
    second_out = torch.empty_like(x)
    third_out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        result = x[row, :] + 10
        if flag > 0:
            first = torch.cumsum(x[row, :] + 1, dim=-1)
            first_out[row, :] = first
            second = torch.cumsum(x[row, :] + 2, dim=-1)
            second_out[row, :] = second
            third = torch.cumsum(x[row, :] + 3, dim=-1)
            third_out[row, :] = third
            if escape:
                result = first + 3
        else:
            first_out[row, :] = -x[row, :]
            second_out[row, :] = -2 * x[row, :]
            third_out[row, :] = -3 * x[row, :]
            if escape:
                result = x[row, :] - 7
        out[row, :] = result
    return out, first_out, second_out, third_out
