"""The fp32 collective-matmul kernels, configs and inputs shared by the CuTe
collective and launch tests: the jagged ``jagged_dense_bmm`` example and a
batched matmul whose operands are computed in the K loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import import_example
import helion
import helion.language as hl

if TYPE_CHECKING:
    from ..runtime.kernel import Kernel

jagged_dense_bmm = import_example("jagged_dense_bmm").jagged_dense_bmm


def _computed_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, rows, reduction = a.shape
    columns = b.size(2)
    out = torch.empty((batches, rows, columns), dtype=a.dtype, device=a.device)
    for batch, row, column in hl.tile([batches, rows, columns]):
        acc = hl.zeros([batch, row, column], dtype=torch.float32)
        for contraction in hl.tile(reduction):
            left = a[batch, row, contraction] * 0.75
            acc = torch.baddbmm(acc, left, b[batch, contraction, column])
        out[batch, row, column] = acc + 0.25
    return out


def collective_kernel(
    *, precision: str = "tf32", batched: bool = False, static_shapes: bool = True
) -> Kernel[torch.Tensor]:
    """The jagged example, or the computed batched matmul when ``batched``."""
    return helion.kernel(
        _computed_batched_matmul if batched else jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        dot_precision=precision,
    )


def collective_config(
    bm: int = 64,
    bn: int = 32,
    bk: int = 64,
    *,
    compute: str = "tcgen05",
    copy: str = "scalar",
) -> helion.Config:
    """A collective-MMA config of either kernel with a ``bm x bn x bk`` tile."""
    return helion.Config(
        block_sizes=[1, bm, bn, bk],
        num_threads=[0, 128 // bn, bn, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy=copy,
    )


def jagged_inputs(
    *, batch: int, rows: int, k: int, n: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    """Uninitialized jagged example inputs whose last group holds every row."""
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=device)
    offsets[-1] = rows
    return (
        offsets,
        torch.empty((rows, k), device=device),
        torch.empty((batch, k, n), device=device),
        torch.empty((batch, n), device=device),
    )


def round_tf32(value: torch.Tensor) -> torch.Tensor:
    """An independent finite-input oracle for ``cvt.rna.tf32.f32``: add half
    a TF32 ULP to the IEEE magnitude, then discard 13 significand bits."""
    bits = value.contiguous().view(torch.int32)
    return ((bits + 0x1000) & -0x2000).view(torch.float32)
