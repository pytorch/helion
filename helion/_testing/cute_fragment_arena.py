"""Written Boolean fragment roots shared by source and runtime tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def sequential_votes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((x.shape[0], 2), dtype=torch.bool, device=x.device)
    masks = torch.empty((x.shape[0], 2, 8, 8), dtype=torch.bool, device=x.device)
    for row in hl.grid(x.shape[0]):
        first = x[row, :, :] > 0
        for _iteration in range(3):
            first = ~first
        masks[row, 0, :, :] = first
        out[row, 0] = torch.any(first)
        second = y[row, :, :] > 0
        for _iteration in range(3):
            second = ~second
        masks[row, 1, :, :] = second
        out[row, 1] = torch.any(second)
    return out, masks


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def explicit_pdl_votes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((x.shape[0], 2), dtype=torch.bool, device=x.device)
    masks = torch.empty((x.shape[0], 2, 8, 8), dtype=torch.bool, device=x.device)
    hl.pdl_wait()
    for row in hl.grid(x.shape[0]):
        first = x[row, :, :] > 0
        for _iteration in range(3):
            first = ~first
        masks[row, 0, :, :] = first
        out[row, 0] = torch.any(first)
        second = y[row, :, :] > 0
        for _iteration in range(3):
            second = ~second
        masks[row, 1, :, :] = second
        out[row, 1] = torch.any(second)
    hl.pdl_launch_dependents()
    return out, masks


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def resident_while_recurrence(
    x: torch.Tensor, limits: torch.Tensor, steps: hl.constexpr, capacity: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    counts = torch.empty((x.size(0),), device=x.device, dtype=torch.int32)
    for row in hl.grid(x.size(0)):
        lane = hl.arange(int(capacity))
        left = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        right = left + 1
        iteration = hl.full([], 0, dtype=torch.int32)
        limit = limits[row]
        while iteration < limit:
            left, right = left + right, left - right
            for _packet in range(steps):
                left = torch.gather(left + 1, 0, ((lane * 3 + 1) % x.size(1)).long())
            iteration = iteration + 1
        hl.store(out, [row, lane], left + right, extra_mask=lane < x.size(1))
        counts[row] = iteration
    return out, counts
