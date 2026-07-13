"""Internal support for DeepGEMM-style grouped BF16 NT benchmarks.

The module backs benchmark and test coverage for the selected CuTe path.
"""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Sequence
import weakref

import torch

import helion
import helion.language as hl

DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M = 64
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_N = 64
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_K = 64
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M = 128
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N = 128
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_K = 64
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_M = 256
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_N = 128
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_K = 64
# DeepGEMM-selected lowering consumes one 224-row source segment per group.
DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_SOURCE_M_TILE = 224
_DEEPGEMM_LAYOUT_VALIDATION_CACHE_MAX = 32
_DEEPGEMM_LAYOUT_VALIDATION_CACHE: OrderedDict[
    tuple[object, ...], tuple[weakref.ReferenceType[torch.Tensor], bool]
] = OrderedDict()
_DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[
        weakref.ReferenceType[torch.Tensor],
        tuple[tuple[int, int, int, int], ...],
        tuple[tuple[int, int], ...],
    ],
] = OrderedDict()
_DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor],
] = OrderedDict()
_DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[
        weakref.ReferenceType[torch.Tensor],
        weakref.ReferenceType[torch.Tensor],
        weakref.ReferenceType[torch.Tensor],
    ],
] = OrderedDict()


def make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
    m_per_group: Sequence[int] = (64, 96, 128, 32),
    *,
    n: int = 64,
    k: int = 64,
    m_alignment: int = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
    tail_padding: int = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
    device: torch.device | str = "cuda",
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Build a DeepGEMM-style contiguous M-grouped BF16 NT problem.

    Rows for each group are packed into ``A_packed`` and followed by ``-1``
    padding rows up to ``m_alignment``. The default device is CUDA to match the
    normal CUDA benchmark default without importing testing helpers into the
    compiler package; pass ``device`` explicitly for CPU-only validation.
    """
    if not m_per_group:
        raise ValueError("m_per_group must contain at least one group")
    if n < 0 or k < 0:
        raise ValueError("n and k must be nonnegative")
    if m_alignment <= 0:
        raise ValueError("m_alignment must be positive")
    if tail_padding < 0:
        raise ValueError("tail_padding must be nonnegative")

    rows: list[torch.Tensor] = []
    layout: list[int] = []
    for group_id, m in enumerate(m_per_group):
        if m < 0:
            raise ValueError("m_per_group values must be nonnegative")
        if m != 0:
            rows.append(torch.randn(m, k, device=device, dtype=torch.bfloat16))
            layout.extend([group_id] * m)
        group_padding = (-m) % m_alignment
        if group_padding != 0:
            rows.append(
                torch.randn(group_padding, k, device=device, dtype=torch.bfloat16)
            )
            layout.extend([-1] * group_padding)

    if tail_padding != 0:
        rows.append(torch.randn(tail_padding, k, device=device, dtype=torch.bfloat16))
        layout.extend([-1] * tail_padding)

    if rows:
        A_packed = torch.cat(rows, dim=0).contiguous()
    else:
        A_packed = torch.empty(0, k, device=device, dtype=torch.bfloat16)
    B_grouped = torch.randn(
        len(m_per_group),
        n,
        k,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    grouped_layout = torch.tensor(layout, device=device, dtype=torch.int32)
    expected = _reference_deepgemm_m_grouped_bf16_gemm_nt_contiguous(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    return (A_packed, B_grouped, grouped_layout), expected


def _reference_deepgemm_m_grouped_bf16_gemm_nt_contiguous(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros(
        A_packed.size(0),
        B_grouped.size(1),
        device=A_packed.device,
        dtype=torch.bfloat16,
    )
    for group_id in range(B_grouped.size(0)):
        rows = grouped_layout == group_id
        if bool(torch.any(rows).item()):
            out[rows] = (A_packed[rows].float() @ B_grouped[group_id].float().T).to(
                out.dtype
            )
    return out


# %%
# DeepGEMM-Style Internal Grouped BF16 Test Harness
# ------------------------------------------


# %%
def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_key(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> tuple[int, ...]:
    return (
        int(A_packed.size(0)),
        int(A_packed.size(1)),
        int(B_grouped.size(0)),
        int(B_grouped.size(1)),
        int(B_grouped.size(2)),
        int(grouped_layout.size(0)),
    )


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel_body(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    """
    Minimal DeepGEMM-style M-grouped contiguous BF16 NT GEMM.

    ``grouped_layout`` marks valid rows with a group id and padding rows with
    ``-1``. Each M tile must contain valid rows for at most one group.
    """
    M_total_aligned, K = A_packed.shape
    _G, N, K2 = B_grouped.shape
    assert K == K2, "K dimension mismatch between A_packed and B_grouped"

    block_m = hl.register_block_size(DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M)
    block_n = hl.register_block_size(DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_N)
    block_k = hl.register_block_size(DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_K)
    out = torch.empty(M_total_aligned, N, dtype=A_packed.dtype, device=A_packed.device)

    for tile_m, tile_n in hl.tile(
        [M_total_aligned, N],
        block_size=[block_m, block_n],
    ):
        group_id = grouped_layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = grouped_layout[tile_m]
        valid_rows = row_group_ids >= 0
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=block_k):
            acc = torch.addmm(
                acc,
                A_packed[tile_m, tile_k],
                B_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid_rows[:, None],  # pyrefly: ignore[bad-index]
            acc.to(out.dtype),
            torch.zeros_like(acc).to(out.dtype),
        )

    return out


_deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel = helion.kernel(
    static_shapes=False,
    key=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_key,
    config=helion.Config(
        block_sizes=[
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_N,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_K,
        ],
    ),
)(_deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel_body)


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_key(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> tuple[int, ...]:
    return (
        int(A_packed.size(0)),
        int(A_packed.size(1)),
        int(B_grouped.size(0)),
        int(B_grouped.size(1)),
        int(B_grouped.size(2)),
        int(work_tile_metadata.size(0)),
    )


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel_body(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    """
    Segment-local generated fallback for DeepGEMM M-grouped BF16 NT.

    ``work_tile_metadata`` is int32 ``[W, 4]`` with rows
    ``(group, global_m_start, valid_m, 0)``. Each row describes one real
    segment-local M tile, so padding-only M tiles are never launched.
    """
    M_total_aligned, K = A_packed.shape
    _G, N, K2 = B_grouped.shape
    assert K == K2, "K dimension mismatch between A_packed and B_grouped"
    assert work_tile_metadata.size(1) == 4

    block_m = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M
    )
    block_n = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N
    )
    block_k = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_K
    )
    out = torch.zeros(
        M_total_aligned,
        N,
        dtype=A_packed.dtype,
        device=A_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), block_m, N],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=block_k):
            a_blk = hl.load(
                A_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                B_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )

    return out


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel_body(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    """
    Full-overwrite generated fallback for DeepGEMM M-grouped BF16 NT.

    This is selected only after host-side layout metadata proves every output
    row is covered by exactly one work tile and no padding rows need preserved
    zeros. Keep the loop/store structure identical to the zero-fill variant.
    """
    M_total_aligned, K = A_packed.shape
    _G, N, K2 = B_grouped.shape
    assert K == K2, "K dimension mismatch between A_packed and B_grouped"
    assert work_tile_metadata.size(1) == 4

    block_m = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M
    )
    block_n = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N
    )
    block_k = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_K
    )
    out = torch.empty(
        M_total_aligned,
        N,
        dtype=A_packed.dtype,
        device=A_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), block_m, N],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=block_k):
            a_blk = hl.load(
                A_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                B_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )

    return out


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel_body(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    """
    DeepGEMM-selected generated segment fallback for BF16 NT.

    ``work_tile_metadata`` is int32 ``[W, 4]`` with compact rows
    ``(group, group_start, actual_m, aligned_m)``. The selected N,M generated
    path derives each 224-row source chunk's valid and store extents on device
    from the grouped scheduler M tile.
    """
    M_total_aligned, K = A_packed.shape
    _G, N, K2 = B_grouped.shape
    assert K == K2, "K dimension mismatch between A_packed and B_grouped"
    assert work_tile_metadata.size(1) == 4

    block_m = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_M
    )
    block_n = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_N
    )
    block_k = hl.register_block_size(
        DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_K
    )
    out = torch.empty(
        M_total_aligned,
        N,
        dtype=A_packed.dtype,
        device=A_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [
            work_tile_metadata.size(0),
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_M,
            N,
        ],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=block_k):
            a_blk = hl.load(
                A_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                B_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )

    return out


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_config() -> helion.Config:
    return helion.Config(
        block_sizes=[
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_K,
        ],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_grouped_static_persistent=True,
        tcgen05_grouped_dynamic_ab_tensormaps=True,
        tcgen05_persistence_model="static_persistent",
        tcgen05_ab_stages=3,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_num_epi_warps=4,
    )


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_config() -> (
    helion.Config
):
    return helion.Config(
        block_sizes=[
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_M,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_N,
            DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_K,
        ],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=7,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_static_persistent=True,
        tcgen05_grouped_dynamic_ab_tensormaps=True,
        tcgen05_deepgemm_selected=True,
        tcgen05_deepgemm_selected_compact_metadata=True,
        tcgen05_selected_accumulator_view="nm",
        tcgen05_selected_d_store_view="nm_transposed",
    )


_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel = helion.kernel(
    static_shapes=False,
    key=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_key,
    config=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_config(),
)(_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel_body)


_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel = helion.kernel(
    static_shapes=False,
    key=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_key,
    config=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_config(),
)(_deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel_body)


_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel = (
    helion.kernel(
        static_shapes=False,
        key=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_key,
        config=_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_config(),
    )(_deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel_body)
)


def _deepgemm_layout_validation_cache_key(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
    *,
    block_m: int,
) -> tuple[object, ...]:
    return (
        id(grouped_layout),
        int(grouped_layout.data_ptr()),
        tuple(int(size) for size in grouped_layout.shape),
        tuple(int(stride) for stride in grouped_layout.stride()),
        int(grouped_layout.storage_offset()),
        grouped_layout.device,
        grouped_layout.dtype,
        int(grouped_layout._version),
        int(A_packed.size(0)),
        int(B_grouped.size(0)),
        int(block_m),
    )


def _deepgemm_layout_validation_cache_add(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
    *,
    layout_has_valid_prefix_tiles: bool,
) -> None:
    _DEEPGEMM_LAYOUT_VALIDATION_CACHE[key] = (
        weakref.ref(grouped_layout),
        layout_has_valid_prefix_tiles,
    )
    _DEEPGEMM_LAYOUT_VALIDATION_CACHE.move_to_end(key)
    while (
        len(_DEEPGEMM_LAYOUT_VALIDATION_CACHE) > _DEEPGEMM_LAYOUT_VALIDATION_CACHE_MAX
    ):
        _DEEPGEMM_LAYOUT_VALIDATION_CACHE.popitem(last=False)


def _deepgemm_layout_validation_cache_has_valid_prefix_tiles(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
) -> bool | None:
    cached_entry = _DEEPGEMM_LAYOUT_VALIDATION_CACHE.get(key)
    if cached_entry is None:
        return None
    cached_ref, layout_has_valid_prefix_tiles = cached_entry
    if cached_ref() is grouped_layout:
        if _deepgemm_cutedsl_layout_segments_cache_get(key, grouped_layout) is None:
            del _DEEPGEMM_LAYOUT_VALIDATION_CACHE[key]
            return None
        _DEEPGEMM_LAYOUT_VALIDATION_CACHE.move_to_end(key)
        return layout_has_valid_prefix_tiles
    del _DEEPGEMM_LAYOUT_VALIDATION_CACHE[key]
    return None


def _deepgemm_cutedsl_layout_segments_cache_add(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
    *,
    segments: tuple[tuple[int, int, int, int], ...],
    padding: tuple[tuple[int, int], ...],
) -> None:
    _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE[key] = (
        weakref.ref(grouped_layout),
        segments,
        padding,
    )
    _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE.move_to_end(key)
    while (
        len(_DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE)
        > _DEEPGEMM_LAYOUT_VALIDATION_CACHE_MAX
    ):
        _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE.popitem(last=False)


def _deepgemm_cutedsl_layout_segments_cache_get(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int], ...]] | None:
    cached_entry = _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE.get(key)
    if cached_entry is None:
        return None
    cached_ref, segments, padding = cached_entry
    if cached_ref() is grouped_layout:
        _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE.move_to_end(key)
        return segments, padding
    del _DEEPGEMM_CUTEDSL_LAYOUT_SEGMENTS_CACHE[key]
    return None


def _deepgemm_cutedsl_segment_tensor_cache_add(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
    *,
    work_tile_metadata: torch.Tensor,
) -> None:
    _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE[key] = (
        weakref.ref(grouped_layout),
        work_tile_metadata,
    )
    _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE.move_to_end(key)
    while (
        len(_DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE)
        > _DEEPGEMM_LAYOUT_VALIDATION_CACHE_MAX
    ):
        _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE.popitem(last=False)


def _deepgemm_cutedsl_segment_tensor_cache_get(
    key: tuple[object, ...],
    grouped_layout: torch.Tensor,
) -> torch.Tensor | None:
    cached_entry = _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE.get(key)
    if cached_entry is None:
        return None
    cached_ref, work_tile_metadata = cached_entry
    if cached_ref() is grouped_layout:
        _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE.move_to_end(key)
        return work_tile_metadata
    del _DEEPGEMM_CUTEDSL_SEGMENT_TENSOR_CACHE[key]
    return None


def _deepgemm_selected_segment_runtime_cache_key(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> tuple[object, ...]:
    return (
        id(A_packed),
        int(A_packed.data_ptr()),
        tuple(int(size) for size in A_packed.shape),
        tuple(int(stride) for stride in A_packed.stride()),
        int(A_packed.storage_offset()),
        A_packed.device,
        A_packed.dtype,
        id(B_grouped),
        int(B_grouped.data_ptr()),
        tuple(int(size) for size in B_grouped.shape),
        tuple(int(stride) for stride in B_grouped.stride()),
        int(B_grouped.storage_offset()),
        B_grouped.device,
        B_grouped.dtype,
        int(B_grouped._version),
        id(grouped_layout),
        int(grouped_layout.data_ptr()),
        tuple(int(size) for size in grouped_layout.shape),
        tuple(int(stride) for stride in grouped_layout.stride()),
        int(grouped_layout.storage_offset()),
        grouped_layout.device,
        grouped_layout.dtype,
        int(grouped_layout._version),
    )


def _deepgemm_selected_segment_runtime_unsupported_cache_add(
    key: tuple[object, ...],
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> None:
    _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE[key] = (
        weakref.ref(A_packed),
        weakref.ref(B_grouped),
        weakref.ref(grouped_layout),
    )
    _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE.move_to_end(key)
    while (
        len(_DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE)
        > _DEEPGEMM_LAYOUT_VALIDATION_CACHE_MAX
    ):
        _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE.popitem(last=False)


def _deepgemm_selected_segment_runtime_unsupported_cache_has(
    key: tuple[object, ...],
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> bool:
    cached_entry = _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE.get(key)
    if cached_entry is None:
        return False
    a_ref, b_ref, layout_ref = cached_entry
    if a_ref() is A_packed and b_ref() is B_grouped and layout_ref() is grouped_layout:
        _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE.move_to_end(key)
        return True
    del _DEEPGEMM_CUTEDSL_SELECTED_SEGMENT_UNSUPPORTED_CACHE[key]
    return False


def _deepgemm_layout_has_valid_prefix_tiles(
    layout: Sequence[int],
    *,
    rows: int,
    block_m: int,
) -> bool:
    for tile_start in range(0, rows, block_m):
        tile_ids = layout[tile_start : tile_start + block_m]
        valid_ids = [group_id for group_id in tile_ids if group_id >= 0]
        if not valid_ids:
            continue
        first_id = tile_ids[0]
        if first_id < 0 or any(group_id != first_id for group_id in valid_ids):
            return False
        seen_padding = False
        for row_group_id in tile_ids:
            if row_group_id < 0:
                seen_padding = True
            elif seen_padding:
                return False
    return True


def _parse_deepgemm_m_grouped_layout_segments(
    grouped_layout: torch.Tensor,
    *,
    group_count: int,
    block_m: int,
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int], ...],
    bool,
]:
    layout = [int(value) for value in grouped_layout.detach().cpu().tolist()]
    segments: list[tuple[int, int, int, int]] = []
    padding: list[tuple[int, int]] = []
    pos = 0
    last_group = -1
    rows = len(layout)
    while pos < rows:
        value = layout[pos]
        if value == -1:
            padding_start = pos
            while pos < rows and layout[pos] == -1:
                pos += 1
            padding.append((padding_start, pos))
            continue
        if value < -1 or value >= group_count:
            raise ValueError(
                "grouped_layout values must be -1 or a group id in [0, G); "
                f"row {pos} has {value}"
            )

        group = value
        if group <= last_group:
            raise ValueError(
                "grouped_layout valid group segments must be strictly "
                f"increasing; row {pos} starts group {group} after group "
                f"{last_group}"
            )
        segment_start = pos
        while pos < rows and layout[pos] == group:
            pos += 1
        actual_m = pos - segment_start
        padding_start = pos
        while pos < rows and layout[pos] == -1:
            pos += 1
        aligned_m = pos - segment_start
        segments.append((group, segment_start, actual_m, aligned_m))
        if padding_start < pos:
            padding.append((padding_start, pos))
        last_group = group

    return (
        tuple(segments),
        tuple(padding),
        _deepgemm_layout_has_valid_prefix_tiles(
            layout,
            rows=rows,
            block_m=block_m,
        ),
    )


def _deepgemm_cuda_graph_capture_active(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
    *,
    block_m: int = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
) -> bool:
    if A_packed.ndim != 2:
        raise ValueError("A_packed must have shape [M_total_aligned, K]")
    if B_grouped.ndim != 3:
        raise ValueError("B_grouped must have shape [G, N, K]")
    if grouped_layout.ndim != 1:
        raise ValueError("grouped_layout must have shape [M_total_aligned]")
    if A_packed.dtype != torch.bfloat16 or B_grouped.dtype != torch.bfloat16:
        raise ValueError("A_packed and B_grouped must be torch.bfloat16")
    if grouped_layout.dtype != torch.int32:
        raise ValueError("grouped_layout must have dtype torch.int32")
    if not A_packed.is_contiguous():
        raise ValueError("A_packed must be contiguous")
    if not B_grouped.is_contiguous():
        raise ValueError("B_grouped must be contiguous")
    if grouped_layout.size(0) != A_packed.size(0):
        raise ValueError("grouped_layout must have one entry per packed A row")
    if A_packed.size(1) != B_grouped.size(2):
        raise ValueError("K dimension mismatch between A_packed and B_grouped")
    if B_grouped.size(0) == 0:
        raise ValueError("B_grouped must contain at least one group")
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    cache_key = _deepgemm_layout_validation_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
        block_m=block_m,
    )
    cached_layout_has_valid_prefix_tiles = (
        _deepgemm_layout_validation_cache_has_valid_prefix_tiles(
            cache_key,
            grouped_layout,
        )
    )
    if cached_layout_has_valid_prefix_tiles is not None:
        return cached_layout_has_valid_prefix_tiles
    if _deepgemm_cuda_graph_capture_active(grouped_layout):
        raise ValueError(
            "grouped_layout content validation is not cached for this tensor; "
            "call deepgemm_m_grouped_bf16_gemm_nt_contiguous once before CUDA "
            "graph capture"
        )
    segments, padding, layout_has_valid_prefix_tiles = (
        _parse_deepgemm_m_grouped_layout_segments(
            grouped_layout,
            group_count=int(B_grouped.size(0)),
            block_m=block_m,
        )
    )
    _deepgemm_cutedsl_layout_segments_cache_add(
        cache_key,
        grouped_layout,
        segments=segments,
        padding=padding,
    )
    _deepgemm_layout_validation_cache_add(
        cache_key,
        grouped_layout,
        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
    )
    return layout_has_valid_prefix_tiles


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[tuple[int, int], ...]] | None:
    cache_key = _deepgemm_layout_validation_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
        block_m=DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
    )
    cached = _deepgemm_cutedsl_layout_segments_cache_get(
        cache_key,
        grouped_layout,
    )
    if cached is not None:
        return cached
    if _deepgemm_cuda_graph_capture_active(grouped_layout):
        return None

    segments, padding, _layout_has_valid_prefix_tiles = (
        _parse_deepgemm_m_grouped_layout_segments(
            grouped_layout,
            group_count=int(B_grouped.size(0)),
            block_m=DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
        )
    )
    _deepgemm_cutedsl_layout_segments_cache_add(
        cache_key,
        grouped_layout,
        segments=segments,
        padding=padding,
    )
    return segments, padding


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor | None:
    cache_key = _deepgemm_layout_validation_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
        block_m=DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M,
    )
    cached = _deepgemm_cutedsl_segment_tensor_cache_get(cache_key, grouped_layout)
    if cached is not None:
        return cached
    if _deepgemm_cuda_graph_capture_active(grouped_layout):
        return None

    metadata = _deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    if metadata is None:
        return None

    segments, _padding = metadata
    block_m = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_M
    work_tiles: list[tuple[int, int, int, int]] = []
    for group, start, actual_m, _aligned_m in segments:
        for tile_start in range(0, actual_m, block_m):
            valid_m = min(block_m, actual_m - tile_start)
            work_tiles.append((group, start + tile_start, valid_m, 0))

    if work_tiles:
        work_tile_metadata = torch.tensor(
            work_tiles,
            device=A_packed.device,
            dtype=torch.int32,
        )
    else:
        work_tile_metadata = torch.empty(
            (0, 4),
            device=A_packed.device,
            dtype=torch.int32,
        )
    _deepgemm_cutedsl_segment_tensor_cache_add(
        cache_key,
        grouped_layout,
        work_tile_metadata=work_tile_metadata,
    )
    return work_tile_metadata


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor | None:
    source_m_tile = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_SOURCE_M_TILE
    cache_key = _deepgemm_layout_validation_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
        block_m=source_m_tile,
    )
    cached = _deepgemm_cutedsl_segment_tensor_cache_get(cache_key, grouped_layout)
    if cached is not None:
        return cached
    if _deepgemm_cuda_graph_capture_active(grouped_layout):
        return None

    metadata = _deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    if metadata is None:
        return None

    segments, _padding = metadata
    work_tiles: list[tuple[int, int, int, int]] = []
    next_row = 0
    for expected_group, (group, start, actual_m, aligned_m) in enumerate(segments):
        if group != expected_group:
            return None
        if start != next_row:
            return None
        if actual_m <= 0 or actual_m > aligned_m:
            return None
        if start % source_m_tile != 0 or aligned_m % source_m_tile != 0:
            return None
        work_tiles.append((group, start, actual_m, aligned_m))
        next_row = start + aligned_m
    if next_row != int(A_packed.size(0)):
        return None

    if work_tiles:
        work_tile_metadata = torch.tensor(
            work_tiles,
            device=A_packed.device,
            dtype=torch.int32,
        )
    else:
        work_tile_metadata = torch.empty(
            (0, 4),
            device=A_packed.device,
            dtype=torch.int32,
        )
    _deepgemm_cutedsl_segment_tensor_cache_add(
        cache_key,
        grouped_layout,
        work_tile_metadata=work_tile_metadata,
    )
    return work_tile_metadata


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> bool:
    metadata = _deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    if metadata is None:
        return False

    segments, padding = metadata
    if padding or not segments:
        return False
    if (
        int(B_grouped.size(1))
        % DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N
        != 0
    ):
        return False

    next_row = 0
    for _group, start, actual_m, aligned_m in segments:
        if start != next_row or actual_m != aligned_m:
            return False
        next_row = start + actual_m
    return next_row == int(A_packed.size(0))


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
    *,
    layout_has_valid_prefix_tiles: bool,
) -> bool:
    if layout_has_valid_prefix_tiles and not (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    ):
        return False
    return (
        os.environ.get("HELION_BACKEND") == "cute"
        and A_packed.device.type == "cuda"
        and B_grouped.device == A_packed.device
        and grouped_layout.device == A_packed.device
        and A_packed.ndim == 2
        and B_grouped.ndim == 3
        and grouped_layout.ndim == 1
        and A_packed.dtype == torch.bfloat16
        and B_grouped.dtype == torch.bfloat16
        and grouped_layout.dtype == torch.int32
        and A_packed.is_contiguous()
        and B_grouped.is_contiguous()
        and grouped_layout.is_contiguous()
        and int(A_packed.size(1)) == int(B_grouped.size(2))
        and int(B_grouped.size(1))
        % DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_N
        == 0
        and int(A_packed.size(1))
        % DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SEGMENT_TILE_K
        == 0
        and int(grouped_layout.size(0)) == int(A_packed.size(0))
        and _deepgemm_m_grouped_bf16_gemm_nt_contiguous_cutedsl_metadata(
            A_packed,
            B_grouped,
            grouped_layout,
        )
        is not None
    )


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> bool:
    if (
        int(B_grouped.size(1))
        % DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_N
        != 0
        or int(A_packed.size(1))
        % DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_SELECTED_TILE_K
        != 0
    ):
        return False
    if (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            A_packed,
            B_grouped,
            grouped_layout,
        )
        is None
    ):
        return False
    selected_cache_key = _deepgemm_selected_segment_runtime_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    return not _deepgemm_selected_segment_runtime_unsupported_cache_has(
        selected_cache_key,
        A_packed,
        B_grouped,
        grouped_layout,
    )


def _deepgemm_m_grouped_bf16_gemm_nt_contiguous_generated_segment(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor | None:
    selected_work_tile_metadata = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )
    selected_cache_key = _deepgemm_selected_segment_runtime_cache_key(
        A_packed,
        B_grouped,
        grouped_layout,
    )
    if selected_work_tile_metadata is not None and not (
        _deepgemm_selected_segment_runtime_unsupported_cache_has(
            selected_cache_key,
            A_packed,
            B_grouped,
            grouped_layout,
        )
    ):
        if int(selected_work_tile_metadata.size(0)) == 0:
            return torch.zeros(
                A_packed.size(0),
                B_grouped.size(1),
                dtype=A_packed.dtype,
                device=A_packed.device,
            )
        selected_args = (
            A_packed,
            B_grouped,
            selected_work_tile_metadata,
        )
        try:
            bound = _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_kernel.bind(
                selected_args
            )
            bound.env.config_spec.cute_tcgen05_search_enabled = True
            return bound(*selected_args)
        except helion.exc.BackendUnsupported:
            _deepgemm_selected_segment_runtime_unsupported_cache_add(
                selected_cache_key,
                A_packed,
                B_grouped,
                grouped_layout,
            )

    work_tile_metadata = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_metadata_tensor(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )
    if work_tile_metadata is None:
        return None
    if int(work_tile_metadata.size(0)) == 0:
        return torch.zeros(
            A_packed.size(0),
            B_grouped.size(1),
            dtype=A_packed.dtype,
            device=A_packed.device,
        )
    segment_args = (
        A_packed,
        B_grouped,
        work_tile_metadata,
    )
    segment_kernel = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_full_overwrite_kernel
        if _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_output_fully_overwritten(
            A_packed,
            B_grouped,
            grouped_layout,
        )
        else _deepgemm_m_grouped_bf16_gemm_nt_contiguous_segment_kernel
    )
    bound = segment_kernel.bind(segment_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    return bound(*segment_args)


def deepgemm_m_grouped_bf16_gemm_nt_contiguous(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    """
    Run the internal DeepGEMM-style M-grouped contiguous BF16 NT entry point
    used by benchmarks and tests.

    Contract:
    - ``A_packed`` is contiguous BF16 ``[M_total_aligned, K]``.
    - ``B_grouped`` is contiguous BF16 ``[G, N, K]`` and computes
      ``A @ B[group].T``.
    - ``grouped_layout`` is int32 ``[M_total_aligned]`` with group ids for
      valid rows and ``-1`` for padding rows.
    - Output is BF16 ``[M_total_aligned, N]``; padding rows are zeroed.
    """
    layout_has_valid_prefix_tiles = (
        _validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )

    if _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
        A_packed,
        B_grouped,
        grouped_layout,
        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
    ):
        generated_result = (
            _deepgemm_m_grouped_bf16_gemm_nt_contiguous_generated_segment(
                A_packed,
                B_grouped,
                grouped_layout,
            )
        )
        if generated_result is not None:
            return generated_result

    if not layout_has_valid_prefix_tiles:
        raise ValueError(
            "deepgemm_m_grouped_bf16_gemm_nt_contiguous mixed-boundary layouts "
            "require the CuTe generated segment fallback"
        )

    return _deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel(
        A_packed,
        B_grouped,
        grouped_layout,
    )
