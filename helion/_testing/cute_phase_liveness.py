"""Ordered phases whose consumer keeps full static tensor axes.

Flat PID scheduling projects a consumer into its own kernel. Persistent
scheduling makes the producer and consumer share the physical block.
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


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def routed_after_producer(
    x: torch.Tensor, payload: torch.Tensor, k: int
) -> torch.Tensor:
    rows, columns = x.shape
    columns = hl.specialize(columns)
    k = hl.specialize(k)
    scores = torch.empty_like(x)
    out = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    for row, column in hl.tile([rows, columns], block_size=[1, None]):
        scores[row, column] = x[row, column] * 3 - 1
    for row, slot in hl.tile([rows, k], block_size=[1, 1]):
        values, indices = torch.topk(scores[row.begin, :], k)
        pick = hl.arange(k) == slot.begin
        index = torch.where(pick, indices, 0).sum()
        selected = torch.where(pick, values, 0).sum()
        out[row.begin, slot.begin] = payload[index] + selected
    return out


def routed_phase_config(block: int, *, warp: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[block],
        pid_type="persistent_blocked",
        cross_loop_pipeline="barrier",
        num_sm_multiplier=1,
        cute_fragment_topk_network=warp,
        cute_fragment_reduction="warp" if warp else "serial",
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def grouped_selection_after_producer(
    x: torch.Tensor, payload: torch.Tensor, out: torch.Tensor, mode: hl.constexpr
) -> torch.Tensor:
    rows, columns = x.shape
    width = columns // 2
    scores = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        scores[row, col] = x[row, col] - 2
    for row, group in hl.tile([rows, 2], block_size=[1, 1]):
        start = group.begin * width
        ordinals = start + hl.arange(width)
        if mode == "negative_gather" or mode == "wrapped_gather":
            ordinals = ordinals - columns
        if mode == "wrapped_gather":
            ordinals = torch.where(ordinals < 0, ordinals + columns, ordinals)
        values, indices = torch.topk(scores[row.begin, ordinals], 4)
        if mode == "negative_gather" or mode == "wrapped_gather":
            extra = torch.gather(payload[ordinals], 0, indices)
        elif mode == "reduced_index" or mode == "nested_reduced_index":
            chosen = torch.where(hl.arange(4) == group.begin, indices, 0).sum()
            extra = payload[chosen + start]
            if mode == "nested_reduced_index":
                for step in range(2):
                    extra = extra + payload[start + step]
        else:
            extra = payload[indices + start]
        out[row.begin, group.begin, :] = values + extra
    return out


@helion.kernel(backend="cute", static_shapes=True)
def explicit_barrier_row_topk(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    scores = torch.empty_like(x)
    out = torch.empty((rows, 4), dtype=x.dtype, device=x.device)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        scores[row, col] = x[row, col] * 3 - 1
    hl.barrier()
    for row, slot in hl.tile([rows, 4], block_size=[1, 1]):
        values = torch.topk(scores[row.begin, :], 4).values
        out[row.begin, slot.begin] = torch.where(
            hl.arange(4) == slot.begin, values, 0
        ).sum()
    return out


@helion.kernel(backend="cute", static_shapes=True)
def explicit_direct_selection(
    x: torch.Tensor,
    payload: torch.Tensor,
    out: torch.Tensor,
    companion: hl.constexpr,
) -> torch.Tensor:
    rows, columns = x.shape
    width = columns // 2
    scores = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        scores[row, col] = x[row, col] - 2
    hl.barrier()
    for row, group in hl.tile([rows, 2], block_size=[1, 1]):
        start = group.begin * width
        values, indices = torch.topk(scores[row.begin, start + hl.arange(width)], 4)
        gathered = values + payload[indices + start]
        if companion == "reduce":
            gathered = gathered + payload[hl.arange(64) % columns].sum()
        elif companion == "loop":
            for _ in hl.grid(3):
                gathered = gathered + 1
        out[row.begin, group.begin, :] = gathered
    return out
