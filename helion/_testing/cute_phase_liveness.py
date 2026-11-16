"""Ordered phases whose consumer keeps full static tensor axes.

Tests select flat PID scheduling to project the barrier-separated consumer
into its own kernel; a persistent launch would not exercise stage liveness.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def slice_sum_phases(
    x: torch.Tensor, y: torch.Tensor, tensor_indices: hl.constexpr
) -> torch.Tensor:
    batches, steps, rows, contraction = x.shape
    cols = y.shape[-1]
    temporary = torch.empty(
        (batches, steps, rows, cols), device=x.device, dtype=torch.float32
    )
    output = torch.empty((batches, rows, cols), device=x.device, dtype=torch.float32)
    for batch, step in hl.grid([batches, steps]):
        r, k, c = hl.arange(rows), hl.arange(contraction), hl.arange(cols)
        value = hl.dot(x[batch, step, r, k], y[batch, step, k, c])
        temporary[batch, step, r, c] = (
            value + torch.sum(value, dim=0)[None, :] + torch.sum(value, dim=1)[:, None]
        )
    hl.barrier()
    for batch in hl.grid(batches):
        result = hl.zeros((rows, cols), dtype=torch.float32)
        row = hl.arange(rows)
        column = hl.arange(cols)
        for consumer_step in hl.grid(steps):
            if tensor_indices:
                result += temporary[batch, consumer_step, row, column]
            else:
                result += temporary[batch, consumer_step, :, :]
        output[batch, row, column] = result
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def padded_factory_phases(x: torch.Tensor) -> torch.Tensor:
    temporary = torch.empty_like(x)
    output = torch.empty((7, 11), device=x.device, dtype=x.dtype)
    for _producer in hl.grid(1):
        temporary[:, :] = x[:, :] + 1
    hl.barrier()
    for _consumer in hl.grid(1):
        factory = hl.zeros((7, 11), dtype=x.dtype)
        output[:, :] = factory + temporary[0, 0]
    return output
