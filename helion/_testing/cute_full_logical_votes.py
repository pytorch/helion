"""Small authored programs for complete computed logical reductions."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def shared_full_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        output[row.id] = torch.any(values > 2) | torch.all(values >= 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def dot_full_vote(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty((x.size(0), x.size(1)), dtype=torch.float32, device=x.device)
    flags = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        scores = hl.dot(x[row.id, :, :], weight[:, :])
        difference = hl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=(
                scores,
                torch.full_like(scores, 0.125),
                torch.full_like(scores, -1.0),
            ),
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        vote = torch.any(difference > 2.0)
        flags[row.id] = vote
        if vote:
            output[row.id, :] = torch.amax(difference, dim=1)
        else:
            output[row.id, :] = 0.0
    return output, flags


@helion.kernel(backend="cute", static_shapes=True)
def two_phase_full_votes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    second = torch.empty((y.size(0),), dtype=torch.bool, device=y.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        first[row.id] = torch.any(values > 2) | torch.all(values >= 0)
    for row in hl.tile(y.size(0), block_size=1):
        values = y[row.id, :, :]
        second[row.id] = torch.any(values > 2) | torch.all(values >= 0)
    return first, second


@helion.kernel(backend="cute", static_shapes=True)
def barrier_phase_full_votes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    second = torch.empty((y.size(0),), dtype=torch.bool, device=y.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        first[row.id] = torch.any(values > 2) | torch.all(values >= 0)
    hl.barrier()
    for row in hl.tile(y.size(0), block_size=1):
        values = y[row.id, :, :]
        second[row.id] = torch.any(values > 2) | torch.all(values >= 0)
    return first, second


@helion.kernel(backend="cute", static_shapes=True)
def output_argument_full_votes(
    x: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        left[row.id, :] = torch.amax(values, dim=1)
        right[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return left, right


@helion.kernel(backend="cute", static_shapes=True)
def grid_axis_full_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = x[rows, :, :]
        output[rows] = torch.any(values > 2) | torch.all(values >= 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def reshaped_grid_axis_full_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = x[rows, :, :].reshape(-1, x.size(2))
        output[rows] = torch.any(values > 2) | torch.all(values > 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def loop_axis_full_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        for keys in hl.tile(x.size(1), block_size=2):
            values = x[row.id, keys, :]
            output[row.id, keys] = torch.any(values > 2) | torch.all(values > 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def literal_reshape_grid_axis_full_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = x[rows, :, :].reshape(64, 32)
        output[rows] = torch.any(values > 2) | torch.all(values > 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def captured_reshape_grid_axis_full_votes(
    x: torch.Tensor, flags: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = x[rows, :, :].reshape(64, 32)
        if flags[rows.id]:
            output[rows] = torch.any(values > 2) | torch.all(values > 0)
        else:
            output[rows] = False
    return output


@helion.kernel(backend="cute", static_shapes=True)
def captured_full_votes(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :].reshape(32, 32)
        if flags[row.id]:
            output[row.id] = torch.any(values > 2) | torch.all(values > 0)
        else:
            output[row.id] = False
    return output


@helion.kernel(backend="cute", static_shapes=True)
def carried_tile_domain_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = hl.zeros((64, 32), dtype=torch.float32)
        output[row.id] = False
        for keys in hl.tile(x.size(1), block_size=2):
            output[row.id] = torch.any(values > 2) | torch.all(values > 0)
            values = x[row.id, keys, :, :].reshape(64, 32)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def carried_complete_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = hl.zeros((32, 32), dtype=torch.float32)
        output[row.id] = False
        for keys in hl.tile(x.size(1), block_size=1):
            output[row.id] = torch.any(values > 2) | torch.all(values > 0)
            values = values + x[row.id, keys.id, :, :]
    return output


@helion.kernel(backend="cute", static_shapes=True)
def conditional_result_tile_votes(
    x: torch.Tensor, flags: torch.Tensor, tile_in_then: hl.constexpr
) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = hl.zeros((64, 32), dtype=torch.float32)
        if flags[rows.id]:
            if tile_in_then:
                values = x[rows, :, :].reshape(64, 32)
            else:
                values = values + 1
        else:
            if tile_in_then:
                values = values + 1
            else:
                values = x[rows, :, :].reshape(64, 32)
        output[rows] = torch.any(values > 2) | torch.all(values > 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def initial_tile_domain_votes(x: torch.Tensor, steps: int) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = x[rows, :, :].reshape(64, 32)
        for _step in hl.tile(steps, block_size=1):
            values = hl.zeros((64, 32), dtype=torch.float32)
        output[rows] = torch.any(values > 2) | torch.all(values > 0)
    return output


@helion.kernel(backend="cute", static_shapes=True)
def while_carried_tile_votes(x: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = hl.zeros((64, 32), dtype=torch.float32)
        iteration = hl.full([], 0, dtype=torch.int32)
        output[rows] = False
        while iteration < limits[rows.id]:
            output[rows] = torch.any(values > 2) | torch.all(values > 0)
            values = x[rows, :, :].reshape(64, 32)
            iteration = iteration + 1
    return output


@helion.kernel(backend="cute", static_shapes=True)
def while_condition_tile_votes(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0), block_size=2):
        values = hl.zeros((64, 32), dtype=torch.float32)
        while torch.any(values > 2) | torch.all(values > 0):
            values = x[rows, :, :].reshape(64, 32)
        output[rows] = False
    return output
