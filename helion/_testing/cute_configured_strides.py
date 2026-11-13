"""A stable matmul result whose output storage pitch is configurable."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def pitched_config(mode: str, pitch: int) -> helion.Config:
    return helion.Config(
        block_sizes=[pitch, 64, 16],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def pitched_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    groups, rows, _inner = x.shape
    cols = y.size(2)
    pitch = hl.register_block_size(rows, rows * 4)
    backing = torch.empty((groups, pitch, cols), device=x.device, dtype=torch.float32)
    out = backing.as_strided((groups, rows, cols), (pitch * cols, cols, 1))
    for group, row, col in hl.tile([groups, rows, cols], block_size=[1, None, None]):
        total = hl.zeros([row, col], dtype=torch.float32)
        for _repeat in hl.grid(2):
            total = hl.dot(x[group.id, row, :], y[group.id, :, col], acc=total)
        out[group.id, row, col] = total * 0.5
    return out
