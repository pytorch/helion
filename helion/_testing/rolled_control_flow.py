from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(autotune_effort="none")
def else_only_carried_reduction(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[1], block_size=1):
        value = x[0, row, :] * 0
        take_positive = row.begin == 0
        for k in hl.tile(x.shape[0], block_size=1):
            if take_positive:
                pass
            else:
                value = value + x[k, row, :].sum(0)
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@helion.kernel(autotune_effort="none")
def carried_reduction(x: torch.Tensor, conditional: hl.constexpr) -> torch.Tensor:
    out = torch.empty((x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.shape[1], block_size=1):
        value = x[0, row, :] * 0
        take_positive = row.begin == 0
        for k in hl.tile(x.shape[0], block_size=1):
            if conditional:
                if take_positive:
                    value = value + x[k, row, :].sum(0)
                else:
                    value = value - x[k, row, :].sum(0)
            else:
                value = value + x[k, row, :].sum(0)
        out[row, :] = value - value.sum(-1, keepdim=True)
    return out


@helion.kernel(autotune_effort="none")
def two_axis_carried_reductions(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out_x = torch.empty((x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
    out_y = torch.empty((y.shape[1], y.shape[2]), dtype=y.dtype, device=y.device)
    for row in hl.tile(x.shape[1], block_size=1):
        value = x[0, row, :] * 0
        for k in hl.tile(x.shape[0], block_size=1):
            value = value + x[k, row, :].sum(0)
        out_x[row, :] = value - value.sum(-1, keepdim=True)
    for row in hl.tile(y.shape[1], block_size=1):
        value = y[0, row, :] * 0
        for k in hl.tile(y.shape[0], block_size=1):
            value = value + y[k, row, :].sum(0)
        out_y[row, :] = value - value.sum(-1, keepdim=True)
    return out_x, out_y
