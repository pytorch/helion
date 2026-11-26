"""Shared CuTe kernels for grouped GEMM seed and code generation tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped_offsets_k64(
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _unsupported_k_loop(
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


def grouped_ab_k_args(
    *,
    m: int = 640,
    k: int = 256,
    n: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    offset_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((4, n, k), dtype=dtype),
        torch.tensor([0, 0, 1, 129, m], dtype=offset_dtype),
    )


def grouped_ab_k_config(cluster_m: int, *, stages: int = 2) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=stages,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_ab_tile_k=128,
    )


def grouped_d_shift_args(kind: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    metadata = (
        [0, 1, 128, 128, 128, 128, 126, 1]
        if kind == "sizes"
        else [0, 0, 1, 129, 257, 385, 513, 639, 640]
    )
    return (
        torch.empty((640, 256), dtype=torch.bfloat16),
        torch.empty((8, 512, 256), dtype=torch.bfloat16),
        torch.tensor(metadata, dtype=dtype),
    )


def grouped_d_shift_config(orientation: str, *, cluster_n: int = 1) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_orientation=orientation,
        tcgen05_grouped_worklist_mma_n=256,
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_full_allocation_a=True,
        tcgen05_grouped_d_shift=True,
    )
