"""Grouped GEMM kernels over device offsets that several grouped test modules bind."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def observed_logical_k(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, int]:
    m, k = a.shape
    groups, n, _ = b.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, b[group, columns, inner].T)
        hl.store(out, [row_index, columns], acc, extra_mask=valid[:, None])
    return out, block_k


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def grouped_offsets_k64(
    a: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    rows, reduction = a.shape
    groups, columns, _ = weights.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((rows, columns), device=a.device, dtype=a.dtype)
    for group_tile, row_tile, column_tile in hl.tile(
        [groups, rows, columns], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        extent = offsets[group + 1] - offsets[group]
        row_indices = start + row_tile.index
        valid = row_tile.index < extent
        acc = hl.zeros([row_tile, column_tile], dtype=torch.float32)
        for inner in hl.tile(reduction, block_size=block_k):
            left = hl.load(a, [row_indices, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, weights[group, column_tile, inner].T)
        hl.store(out, [row_indices, column_tile], acc, extra_mask=valid[:, None])
    return out


def fresh_grouped_offsets_k64() -> Kernel[torch.Tensor]:
    # A fresh bind is needed when comparing seed-generation policies.
    return helion.kernel(
        grouped_offsets_k64.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def unsupported_k_loop(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, variant: int
) -> torch.Tensor:
    hl.specialize(variant)
    m, k = a.shape
    groups, n, _ = b.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.full(
            [rows, columns], 1.0 if variant == 1 else 0.0, dtype=torch.float32
        )
        begin = 64 if variant == 2 else 0
        for inner in hl.tile(begin, k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            if variant == 3:
                left = left * 1.0
            acc = torch.addmm(
                acc,
                left,
                b[group, columns, inner].T,
                alpha=2 if variant == 4 else 1,
            )
            if variant == 5:
                acc = acc * 1.0
        hl.store(out, [row_index, columns], acc, extra_mask=valid[:, None])
    return out
