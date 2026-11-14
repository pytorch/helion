"""Written two-stage materialization for ordered complete-root proofs."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def dependent_full_votes(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2) | torch.all(values >= 0)
        if vote:
            partial[row.id, :, :] = values + 1
        else:
            partial[row.id, :, :] = values - 1
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        for step in range(2):
            values = values + step
        vote = torch.any(values > 0) | torch.all(values < 10)
        if vote:
            output[row.id] = torch.all(values < 12)
        else:
            output[row.id] = torch.any(values < -12)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def dependent_dot_votes(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partial = torch.empty(
        (x.size(0), x.size(1), weight.size(1)),
        dtype=torch.float32,
        device=x.device,
    )
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        scores = hl.dot(x[row.id, :, :], weight[:, :])
        vote = torch.any(scores > 2) | torch.all(scores >= 0)
        if vote:
            partial[row.id, :, :] = scores + 1
        else:
            partial[row.id, :, :] = scores - 1
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def while_dependent_votes(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        while torch.any(values > 2):
            values = values - 1
        partial[row.id, :, :] = values + torch.any(values > 2)
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def explicit_dependent_votes(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        partial[row.id, :, :] = values + torch.any(values > 2)
    hl.barrier()
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def three_dependent_votes(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        partial[row.id, :, :] = values + torch.any(values > 2)
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    for row in hl.tile(x.size(0), block_size=1):
        output[row.id] = ~output[row.id]
    return output


def _opaque_host_value(value: int) -> int:
    return value


@helion.kernel(backend="cute", static_shapes=True)
def opaque_host_dependent_votes(x: torch.Tensor) -> torch.Tensor:
    # Tracing can evaluate this, but its body is not an admitted pure host
    # metadata operation, so the projection must never replay the prelude.
    scale = _opaque_host_value(x.size(0))
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        partial[row.id, :, :] = values * scale + torch.any(values > 2)
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def specialized_dependent_votes(
    x: torch.Tensor, increment: int, other: int, mode: hl.constexpr
) -> torch.Tensor:
    if mode == "identity":
        increment = hl.specialize(increment)
    elif mode == "arithmetic":
        increment = increment + 1
    elif mode == "other":
        increment = hl.specialize(other)
    elif mode == "opaque":
        increment = _opaque_host_value(increment)
    elif mode == "tensor":
        x = torch.empty_like(x)
    partial = torch.empty_like(x)
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        partial[row.id, :, :] = values + increment + torch.any(values > 2)
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0) | torch.all(values < 10)
    return output
