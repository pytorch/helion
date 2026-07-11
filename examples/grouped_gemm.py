"""
Grouped GEMM Example
====================

This example demonstrates grouped matrix multiplication (GEMM) where multiple
input matrices ``A_i`` (with potentially different numbers of rows ``M_i``)
are multiplied against a single shared weight matrix ``B``. The results are
concatenated in the original group order.

Key ideas used in this implementation:

- Pack all groups' rows into one contiguous tensor ``A_packed`` with shape
  ``[sum(M_i), K]``. This improves memory locality and simplifies indexing.

- Represent group boundaries with ``group_offsets`` (size ``G+1``), so that
  rows for group ``g`` live in ``A_packed[group_offsets[g]:group_offsets[g+1]]``.

- Use data-dependent tiling over the concatenated row dimension to efficiently
  support jagged group sizes (different ``M_i`` per group) without padding.

The example provides these grouped GEMM entry points:

1) ``grouped_gemm_jagged`` - a simple kernel that iterates groups and tiles
   dynamically.

2) ``grouped_gemm_jagged_persistent`` - a persistent kernel with dynamic tile
   assignment for better load balancing across streaming multiprocessors (SMs).

3) ``blackwell_grouped_gemm_nt`` - a Blackwell CuTeDSL-style grouped GEMM
   harness with one independent ``A_g @ B_g.T`` problem per group. Groups may
   use different ``M``, ``N``, and ``K`` sizes.

4) ``deepgemm_m_grouped_bf16_gemm_nt_contiguous`` - a DeepGEMM-style
   M-grouped contiguous BF16 NT harness. A is packed as
   ``[M_total_aligned, K]``, B is grouped as contiguous ``[G, N, K]``, and
   ``grouped_layout`` maps packed rows to group ids or ``-1`` padding.

5) ``blackwell_grouped_gemm_nt_direct`` - an explicit opt-in generated
   direct-pointer variant of the Blackwell grouped GEMM harness. Unsupported
   shapes fail instead of falling back.
"""

# %%
# Imports and Dependencies
# ------------------------

# %%
from __future__ import annotations

from collections import OrderedDict
from itertools import starmap
import os
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Sequence
import weakref

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
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
BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES = (
    (128, 128, 128, 1),
    (512, 128, 128, 1),
    (128, 256, 128, 1),
)
BLACKWELL_GROUPED_GEMM_DOC_PROBLEM_SIZES = (
    (8192, 1280, 32, 1),
    (16, 384, 1536, 1),
    (640, 1280, 16, 1),
    (640, 160, 16, 1),
)
_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX = 16
# Reserved-SM caps are scoped to the generated direct Blackwell grouped-GEMM
# path. They are B200-measured tail-wave launch caps; the selector returns 0
# outside those partial-wave bands.
_BLACKWELL_GROUPED_GENERATED_MTAIL_RESERVED_SMS = 3
_BLACKWELL_GROUPED_GENERATED_LARGE_MTAIL_RESERVED_SMS = 2


class _BlackwellGeneratedTensorGuard(NamedTuple):
    tensor_id: int
    data_ptr: int
    device_type: str
    device_index: int | None
    dtype: torch.dtype
    ndim: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class _BlackwellGeneratedStableLaunchGuard(NamedTuple):
    mma_tiler_mn: tuple[int, ...]
    cluster_shape_mn: tuple[int, ...]
    use_2cta_instrs: bool
    a_guards: tuple[_BlackwellGeneratedTensorGuard, ...]
    b_guards: tuple[_BlackwellGeneratedTensorGuard, ...]
    out_guards: tuple[_BlackwellGeneratedTensorGuard, ...]


class _BlackwellGeneratedStableLaunch(NamedTuple):
    cache_key: tuple[object, ...]
    guard: _BlackwellGeneratedStableLaunchGuard
    kernel_args: tuple[torch.Tensor, ...]
    bound: Callable[..., object]
    fast_call: Callable[[], object] | None
    cuda_graph: torch.cuda.CUDAGraph | None
    runtime_cache_entry: Any | None


class _BlackwellGeneratedStableLaunchCache(
    OrderedDict[tuple[object, ...], _BlackwellGeneratedStableLaunch]
):
    def _drop_last_if(
        self,
        launch: object,
    ) -> None:
        global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

        if _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH is launch:
            _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = None

    def __setitem__(
        self,
        key: tuple[object, ...],
        value: _BlackwellGeneratedStableLaunch,
    ) -> None:
        self._drop_last_if(self.get(key))
        super().__setitem__(key, value)

    def __delitem__(self, key: tuple[object, ...]) -> None:
        launch = self[key]
        super().__delitem__(key)
        self._drop_last_if(launch)

    def clear(self) -> None:
        global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

        super().clear()
        _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = None

    def discard(
        self,
        key: tuple[object, ...],
    ) -> None:
        try:
            launch = self[key]
        except KeyError:
            return
        super().__delitem__(key)
        self._drop_last_if(launch)

    def popitem(
        self,
        last: bool = True,
    ) -> tuple[tuple[object, ...], _BlackwellGeneratedStableLaunch]:
        key, launch = super().popitem(last=last)
        self._drop_last_if(launch)
        return key, launch


_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE: _BlackwellGeneratedStableLaunchCache = (
    _BlackwellGeneratedStableLaunchCache()
)
_BLACKWELL_GENERATED_LAST_STABLE_LAUNCH: _BlackwellGeneratedStableLaunch | None = None

# %%
# Grouped GEMM Kernel - Basic Implementation
# ------------------------------------------


# %%
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged(
    A_packed: torch.Tensor,  # [total_M, K], where total_M == sum(M_i)
    B: torch.Tensor,  # [K, N] shared across all groups
    group_offsets: torch.Tensor,  # [G+1], int32/int64, row offsets into A_packed
) -> torch.Tensor:  # [total_M, N] concatenated outputs for all groups
    """
    Perform grouped GEMM on jagged inputs using row offsets.

    Args:
        A_packed: Row-wise concatenation of per-group inputs ``A_i``,
            shape ``[sum(M_i), K]``.
        B: Shared weight matrix, shape ``[K, N]``.
        group_offsets: Row offsets delimiting each group within ``A_packed``,
            shape ``[G+1]``. For group ``g``: rows are
            ``start = group_offsets[g]`` to ``end = group_offsets[g+1]``.

    Returns:
        Output tensor of shape ``[sum(M_i), N]`` equal to
        ``torch.cat([A_i @ B for i in groups], dim=0)``.
    """
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2, "K dimension mismatch between A_packed and B"

    out = torch.empty(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1

    # Process each group independently, tiling over its specific M_g dimension
    for g in hl.grid(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        M_g = end - start
        if M_g != 0:
            # Create 2D tiling pattern over output dimensions (M_g x N) for current group
            for tile_m, tile_n in hl.tile([M_g, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                # K-reduction loop: multiply tiles along K dimension
                for tile_k in hl.tile(K):
                    a_blk = A_packed[start + tile_m.index, tile_k]
                    b_blk = B[tile_k, tile_n]
                    # Perform fused multiply-add with FP32 accumulation for numerical stability
                    acc = torch.addmm(acc, a_blk, b_blk)
                # Convert accumulator to output dtype and store result
                out[start + tile_m.index, tile_n] = acc.to(out.dtype)

    return out


# %%
# Grouped GEMM Kernel - Persistent Implementation
# -----------------------------------------------


# %%
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged_persistent(
    A_packed: torch.Tensor,  # [total_M, K]
    B: torch.Tensor,  # [K, N]
    group_offsets: torch.Tensor,  # [G+1], row offsets into A_packed
) -> torch.Tensor:
    """
    Persistent grouped GEMM with dynamic tile metadata computation.

    This variant computes tile assignments dynamically in the kernel,
    similar to TritonBench's WS variant.

    Args:
        A_packed: Packed A, concatenated by rows across groups, ``[sum(M_i), K]``.
        B: Shared weight matrix, ``[K, N]``.
        group_offsets: Row offsets delimiting each group within ``A_packed``.

    Returns:
        Output tensor of shape ``[sum(M_i), N]``.
    """
    # Set worker count to match the backend's persistent worker abstraction.
    num_workers = helion.runtime.get_num_sm(A_packed.device)

    # Define tunable block sizes for M, N dimensions (auto-tuned at runtime)
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2

    out = torch.zeros(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1

    for worker_id in hl.grid(num_workers):
        # Persistent thread pattern: each worker processes tiles across all groups
        # using strided/interleaved assignment for load balancing.
        # (i.e. each worker takes every num_workers-th tile. e.g., worker 0 takes tiles 0, N, 2N, ...)
        for g in hl.grid(G):
            group_start = group_offsets[g]
            group_end = group_offsets[g + 1]
            m_size = group_end - group_start

            if m_size > 0:
                # Compute tile grid dimensions for current group
                num_m_tiles = (m_size + BLOCK_M - 1) // BLOCK_M
                # Calculate number of N tiles (shared across all groups)
                num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
                num_group_tiles = num_m_tiles * num_n_tiles

                # Distribute tiles among workers using strided access pattern
                for local_tile in hl.grid(num_group_tiles):
                    tile_in_group = local_tile * num_workers + worker_id
                    if tile_in_group < num_group_tiles:
                        # Convert linear tile index to 2D (M, N) tile coordinates
                        # pyrefly: ignore[unsupported-operation]
                        m_tile_idx = tile_in_group % num_m_tiles
                        # pyrefly: ignore[unsupported-operation]
                        n_tile_idx = tile_in_group // num_m_tiles

                        # Compute global memory indices for current tile
                        base_row = group_start + m_tile_idx * BLOCK_M
                        # pyrefly: ignore[unsupported-operation]
                        base_col = n_tile_idx * BLOCK_N

                        # Generate row and column index ranges for tile access
                        row_idx = base_row + hl.arange(BLOCK_M)
                        col_idx = base_col + hl.arange(BLOCK_N)

                        # Apply boundary masks to handle partial tiles at edges
                        rows_valid = row_idx < group_end
                        cols_valid = col_idx < N

                        # Initialize FP32 accumulator for numerical precision
                        acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)

                        # Iterate over K dimension in blocks for matrix multiplication
                        for k_tile in hl.tile(K):
                            k_idx = k_tile.index

                            # Load tiles from A_packed and B with boundary checking
                            a_blk = hl.load(
                                A_packed,
                                [row_idx, k_idx],
                                extra_mask=rows_valid[  # pyrefly: ignore[bad-index]
                                    :, None
                                ],
                            )
                            b_blk = hl.load(
                                B,
                                [k_idx, col_idx],
                                extra_mask=cols_valid[None, :],  # pyrefly: ignore[bad-index]
                            )

                            # Perform tile-level matrix multiplication and accumulate
                            acc = torch.addmm(acc, a_blk, b_blk)

                        # Write accumulated result to output with boundary masking
                        # pyrefly: ignore[bad-index]
                        valid_2d = rows_valid[:, None] & cols_valid[None, :]
                        hl.store(
                            out,
                            [row_idx, col_idx],
                            acc.to(out.dtype),
                            extra_mask=valid_2d,
                        )

    return out


# %%
# Data Preparation Utilities
# --------------------------


# %%
def _pack_group_inputs(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build ``A_packed``, shared ``B``, and ``group_offsets`` from grouped inputs.

    Expectations:
    - All ``A_i`` share the same ``K`` and dtype/device.
    - All groups share the same ``B`` (as produced by TritonBench inputs).

    Returns ``(A_packed, B_shared, group_offsets)`` where
    ``group_offsets`` has length ``G+1`` with ``group_offsets[0] == 0`` and
    ``group_offsets[g+1] - group_offsets[g] == M_g``.
    """
    assert len(group_A) > 0
    device = group_A[0].device
    dtype = group_A[0].dtype

    # Extract shared weight matrix B (same for all groups in TritonBench)
    B_shared = group_B[0]

    # Compute group offsets and concatenate all A matrices row-wise
    M_sizes = [int(a.size(0)) for a in group_A]
    starts = [0]
    for m in M_sizes:
        starts.append(starts[-1] + m)
    group_offsets = torch.tensor(starts, device=device, dtype=torch.int32)
    A_packed = torch.cat(group_A, dim=0).to(device=device, dtype=dtype).contiguous()
    return A_packed, B_shared, group_offsets


# %%
# DeepGEMM-Style Public Grouped BF16 Harness
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


class DeepGemmMGroupedBf16GemmNtContiguousRouteReport(NamedTuple):
    route: str
    layout_has_valid_prefix_tiles: bool
    use_generated_segment_kernel: bool
    use_selected_segment_kernel: bool
    selected_segment_work_tile_metadata: torch.Tensor | None

    @property
    def selected_metadata_nonempty(self) -> bool:
        metadata = self.selected_segment_work_tile_metadata
        return metadata is not None and int(metadata.size(0)) > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "layout_has_valid_prefix_tiles": self.layout_has_valid_prefix_tiles,
            "use_generated_segment_kernel": self.use_generated_segment_kernel,
            "use_selected_segment_kernel": self.use_selected_segment_kernel,
            "selected_metadata_nonempty": self.selected_metadata_nonempty,
            "route": self.route,
        }


def deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> DeepGemmMGroupedBf16GemmNtContiguousRouteReport:
    """
    Report the route selected by the DeepGEMM M-grouped contiguous BF16 NT API.

    The report validates the public API arguments and reflects runtime fallback
    state, so callers can compare route selection before and after a public API
    call without depending on underscored selector helpers.
    """
    layout_has_valid_prefix_tiles = (
        _validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )
    selected_work_tile_metadata = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )
    use_selected_segment_kernel = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
            A_packed,
            B_grouped,
            grouped_layout,
        )
    )
    use_generated_segment_kernel = (
        _deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
            A_packed,
            B_grouped,
            grouped_layout,
            layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
        )
    )
    selected_metadata_nonempty = (
        selected_work_tile_metadata is not None
        and int(selected_work_tile_metadata.size(0)) > 0
    )
    if (
        use_generated_segment_kernel
        and use_selected_segment_kernel
        and selected_metadata_nonempty
    ):
        route = "generated_selected_segment"
    elif use_generated_segment_kernel:
        route = "generated_segment"
    else:
        route = "generic"
    return DeepGemmMGroupedBf16GemmNtContiguousRouteReport(
        route=route,
        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
        use_generated_segment_kernel=use_generated_segment_kernel,
        use_selected_segment_kernel=use_selected_segment_kernel,
        selected_segment_work_tile_metadata=(
            selected_work_tile_metadata.detach().clone()
            if selected_work_tile_metadata is not None
            else None
        ),
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
    Run the DeepGEMM M-grouped contiguous BF16 NT public harness.

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


def make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
    m_per_group: Sequence[int] = (64, 96, 128, 32),
    *,
    n: int = 64,
    k: int = 64,
    m_alignment: int = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
    tail_padding: int = DEEPGEMM_M_GROUPED_BF16_GEMM_NT_CONTIGUOUS_TILE_M,
    device: torch.device | str = DEVICE,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Build a DeepGEMM-style contiguous M-grouped BF16 NT problem.

    Rows for each group are packed into ``A_packed`` and followed by ``-1``
    padding rows up to ``m_alignment``.
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


# %%
# Blackwell CuTeDSL-Style Grouped GEMM Harness
# --------------------------------------------


# %%
class _BlackwellGeneratedGemmProblem(NamedTuple):
    m: int
    n: int
    k: int


class _BlackwellGeneratedGemmMetadata(NamedTuple):
    device: torch.device
    problems: tuple[_BlackwellGeneratedGemmProblem, ...]
    a_mode0_major: bool
    b_mode0_major: bool
    c_mode0_major: bool


def _blackwell_grouped_gemm_nt_generated_major_mode(
    tensor: torch.Tensor,
    *,
    mode0: str,
    mode1: str,
    tensor_name: str,
    group: int,
) -> str:
    if int(tensor.stride(1)) == 1:
        return mode1
    if int(tensor.stride(0)) == 1:
        return mode0
    raise ValueError(
        f"{tensor_name}[{group}] must have a contiguous {mode0} or {mode1} mode"
    )


def _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
    tensor: torch.Tensor,
    *,
    major: str,
    mode0: str,
    tensor_name: str,
    group: int,
) -> None:
    if int(tensor.data_ptr()) % 16 != 0:
        raise ValueError(f"{tensor_name}[{group}] data pointer must be 16-byte aligned")
    elements_per_16b = 16 // tensor.element_size()
    major_dim = 0 if major == mode0 else 1
    if int(tensor.size(major_dim)) % elements_per_16b != 0:
        raise ValueError(
            f"{tensor_name}[{group}] contiguous dimension must be 16-byte aligned"
        )
    leading_stride_bytes = int(tensor.stride(1 - major_dim)) * tensor.element_size()
    if leading_stride_bytes % 16 != 0:
        raise ValueError(
            f"{tensor_name}[{group}] leading stride must be 16-byte aligned"
        )


def _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    group: int,
) -> None:
    for stride in tensor.stride():
        if stride < 0 or stride > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"{tensor_name}[{group}] strides must be non-negative int32 values"
            )


def _blackwell_grouped_gemm_nt_generated_metadata(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
) -> _BlackwellGeneratedGemmMetadata:
    if not a_groups:
        raise ValueError("grouped GEMM requires at least one group")
    if len(a_groups) != len(b_groups) or len(a_groups) != len(out_groups):
        raise ValueError("A, B, and output group counts must match")

    first_a = a_groups[0]
    first_b = b_groups[0]
    first_out = out_groups[0]
    if first_a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "generated grouped GEMM A/B dtype must be torch.float16 or torch.bfloat16"
        )
    if first_out.dtype != first_a.dtype:
        raise ValueError("generated grouped GEMM output dtype must match A/B dtype")
    if first_a.device.type != "cuda":
        raise ValueError("generated grouped GEMM tensors must be on CUDA")
    if first_b.device != first_a.device or first_out.device != first_a.device:
        raise ValueError("all generated grouped GEMM tensors must share one device")

    a_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_a,
        mode0="m",
        mode1="k",
        tensor_name="A",
        group=0,
    )
    b_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_b,
        mode0="n",
        mode1="k",
        tensor_name="B",
        group=0,
    )
    c_major = _blackwell_grouped_gemm_nt_generated_major_mode(
        first_out,
        mode0="m",
        mode1="n",
        tensor_name="out",
        group=0,
    )

    problems: list[_BlackwellGeneratedGemmProblem] = []
    for group, (a, b, out) in enumerate(
        zip(a_groups, b_groups, out_groups, strict=True)
    ):
        if a.ndim != 2:
            raise ValueError(f"A[{group}] must have shape [M, K]")
        if b.ndim != 2:
            raise ValueError(f"B[{group}] must have shape [N, K]")
        if out.ndim != 2:
            raise ValueError(f"out[{group}] must have shape [M, N]")
        if a.dtype != first_a.dtype or b.dtype != first_a.dtype:
            raise ValueError("all generated grouped GEMM A/B tensors must share dtype")
        if out.dtype != first_out.dtype:
            raise ValueError("all generated grouped GEMM outputs must share dtype")
        if a.device != first_a.device or b.device != first_a.device:
            raise ValueError("all generated grouped GEMM A/B tensors must share device")
        if out.device != first_a.device:
            raise ValueError("all generated grouped GEMM outputs must share device")

        m = int(a.size(0))
        k = int(a.size(1))
        n = int(b.size(0))
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError("generated grouped GEMM dimensions must be positive")
        if int(b.size(1)) != k:
            raise ValueError(f"K dimension mismatch for group {group}")
        if tuple(int(size) for size in out.shape) != (m, n):
            raise ValueError(f"out[{group}] must have shape [{m}, {n}]")

        group_a_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            a,
            mode0="m",
            mode1="k",
            tensor_name="A",
            group=group,
        )
        group_b_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            b,
            mode0="n",
            mode1="k",
            tensor_name="B",
            group=group,
        )
        group_c_major = _blackwell_grouped_gemm_nt_generated_major_mode(
            out,
            mode0="m",
            mode1="n",
            tensor_name="out",
            group=group,
        )
        if group_a_major != a_major:
            raise ValueError("all generated grouped GEMM A tensors must share layout")
        if group_b_major != b_major:
            raise ValueError("all generated grouped GEMM B tensors must share layout")
        if group_c_major != c_major:
            raise ValueError("all generated grouped GEMM outputs must share layout")

        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            a,
            major=group_a_major,
            mode0="m",
            tensor_name="A",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            b,
            major=group_b_major,
            mode0="n",
            tensor_name="B",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_16b_alignment(
            out,
            major=group_c_major,
            mode0="m",
            tensor_name="out",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            a,
            tensor_name="A",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            b,
            tensor_name="B",
            group=group,
        )
        _blackwell_grouped_gemm_nt_generated_check_strides_fit_int32(
            out,
            tensor_name="out",
            group=group,
        )
        problems.append(_BlackwellGeneratedGemmProblem(m, n, k))

    return _BlackwellGeneratedGemmMetadata(
        device=first_a.device,
        problems=tuple(problems),
        a_mode0_major=a_major == "m",
        b_mode0_major=b_major == "n",
        c_mode0_major=c_major == "m",
    )


@helion.kernel(backend="cute")
def _blackwell_grouped_gemm_nt_generated_kernel(
    a_placeholder: torch.Tensor,
    b_placeholder: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out_placeholder: torch.Tensor,
    direct_pointers: torch.Tensor,
    direct_strides: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a_placeholder.size()
    _g, max_n, _k = b_placeholder.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a_placeholder[tile_m, tile_k]
            b_tile = b_placeholder[safe_group_id, tile_n, tile_k]
            valid_k_mask = valid_k[None, :]  # pyrefly: ignore[bad-index]
            masked_a = torch.where(valid_k_mask, a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k_mask, b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out_placeholder[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out_placeholder.dtype),
            out_placeholder[tile_m, tile_n],
        )
    return out_placeholder


def _blackwell_grouped_gemm_nt_generated_config(
    *, reserved_sms: int = 0
) -> helion.Config:
    config_kwargs: dict[str, Any] = {}
    if reserved_sms:
        config_kwargs["tcgen05_grouped_static_reserved_sms"] = int(reserved_sms)
    return helion.Config(
        block_sizes=[128, 64, 64],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_static_persistent=True,
        tcgen05_grouped_dynamic_ab_tensormaps=True,
        tcgen05_grouped_direct_pointer_metadata=True,
        tcgen05_grouped_external_direct_pointers="direct_pointers",
        tcgen05_grouped_external_direct_strides="direct_strides",
        **config_kwargs,
    )


def _blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
    problems: Sequence[_BlackwellGeneratedGemmProblem],
    *,
    num_sm: int,
    block_m: int,
    block_n: int,
) -> int:
    if num_sm <= 0 or block_m <= 0 or block_n <= 0:
        return 0
    total_ctas = 0
    m_tail_ctas = 0
    for problem in problems:
        m_tiles = (problem.m + block_m - 1) // block_m
        n_tiles = (problem.n + block_n - 1) // block_n
        total_ctas += m_tiles * n_tiles
        if problem.m % block_m != 0:
            m_tail_ctas += n_tiles
    if m_tail_ctas == 0:
        return 0
    residual_ctas = total_ctas % num_sm
    # The active-cluster cap is only useful for M-tail work landing in a
    # partially full final scheduler wave. Small-tail waves below this band and
    # no-tail shapes retain the default full-SM launch.
    if residual_ctas * 16 > num_sm * 7 and residual_ctas * 2 < num_sm:
        return _BLACKWELL_GROUPED_GENERATED_MTAIL_RESERVED_SMS
    if (
        residual_ctas * 8 >= num_sm * 3
        and residual_ctas * 16 <= num_sm * 7
        and residual_ctas * 2 < num_sm
        and m_tail_ctas * 8 >= num_sm
    ):
        return _BLACKWELL_GROUPED_GENERATED_LARGE_MTAIL_RESERVED_SMS
    return 0


def _blackwell_grouped_gemm_nt_generated_config_block_mn(
    config: helion.Config,
) -> tuple[int, int]:
    block_sizes = config.config["block_sizes"]
    assert isinstance(block_sizes, list)
    return int(block_sizes[0]), int(block_sizes[1])


def _blackwell_grouped_gemm_nt_generated_reserved_sms(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    block_m: int,
    block_n: int,
) -> int:
    if not group_A:
        return 0
    device = group_A[0].device
    if device.type != "cuda":
        return 0
    num_sm = int(torch.cuda.get_device_properties(device).multi_processor_count)
    problems = tuple(
        _BlackwellGeneratedGemmProblem(
            int(a.size(0)),
            int(b.size(0)),
            int(a.size(1)),
        )
        for a, b in zip(group_A, group_B, strict=True)
    )
    return _blackwell_grouped_gemm_nt_generated_problem_reserved_sms(
        problems,
        num_sm=num_sm,
        block_m=block_m,
        block_n=block_n,
    )


def _blackwell_grouped_gemm_nt_generated_capture_active() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _blackwell_grouped_gemm_nt_placeholder(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if len(shape) == 2:
        base = torch.empty(max(1, shape[1]), device=device, dtype=dtype)
        return torch.as_strided(base, shape, (0, 1))
    if len(shape) == 3:
        base = torch.empty(max(1, shape[2]), device=device, dtype=dtype)
        return torch.as_strided(base, shape, (0, 0, 1))
    raise AssertionError(f"unexpected placeholder shape {shape!r}")


def _blackwell_grouped_gemm_nt_generated_tensor_cache_key(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    return (
        id(tensor),
        int(tensor.data_ptr()),
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tensor.ndim,
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        int(tensor.storage_offset()),
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guard(
    tensor: torch.Tensor,
) -> _BlackwellGeneratedTensorGuard:
    return _BlackwellGeneratedTensorGuard(
        tensor_id=id(tensor),
        data_ptr=int(tensor.data_ptr()),
        device_type=tensor.device.type,
        device_index=tensor.device.index,
        dtype=tensor.dtype,
        ndim=tensor.ndim,
        shape=tuple(int(size) for size in tensor.shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_guard(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> _BlackwellGeneratedStableLaunchGuard:
    return _BlackwellGeneratedStableLaunchGuard(
        mma_tiler_mn=tuple(int(dim) for dim in mma_tiler_mn),
        cluster_shape_mn=tuple(int(dim) for dim in cluster_shape_mn),
        use_2cta_instrs=bool(use_2cta_instrs),
        a_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(a) for a in group_A
        ),
        b_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(b) for b in group_B
        ),
        out_guards=tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_guard(out) for out in out_groups
        ),
    )


def _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
    actual: Sequence[int],
    expected: Sequence[int],
) -> bool:
    return len(actual) == len(expected) and all(
        int(actual_value) == expected_value
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guard_matches(
    guard: _BlackwellGeneratedTensorGuard,
    tensor: torch.Tensor,
) -> bool:
    return (
        guard.tensor_id == id(tensor)
        and guard.data_ptr == int(tensor.data_ptr())
        and guard.device_type == tensor.device.type
        and guard.device_index == tensor.device.index
        and guard.dtype == tensor.dtype
        and guard.ndim == tensor.ndim
        and guard.shape == tuple(int(size) for size in tensor.shape)
        and guard.stride == tuple(int(stride) for stride in tensor.stride())
        and guard.storage_offset == int(tensor.storage_offset())
    )


def _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
    guards: Sequence[_BlackwellGeneratedTensorGuard],
    tensors: Sequence[torch.Tensor],
) -> bool:
    return len(guards) == len(tensors) and all(
        starmap(
            _blackwell_grouped_gemm_nt_generated_tensor_guard_matches,
            zip(guards, tensors, strict=True),
        )
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
    guard: _BlackwellGeneratedStableLaunchGuard,
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> bool:
    return (
        _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
            mma_tiler_mn,
            guard.mma_tiler_mn,
        )
        and _blackwell_grouped_gemm_nt_generated_int_sequence_matches(
            cluster_shape_mn,
            guard.cluster_shape_mn,
        )
        and guard.use_2cta_instrs == bool(use_2cta_instrs)
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.a_guards,
            group_A,
        )
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.b_guards,
            group_B,
        )
        and _blackwell_grouped_gemm_nt_generated_tensor_guards_match(
            guard.out_guards,
            out_groups,
        )
    )


def _blackwell_grouped_gemm_nt_generated_stable_launch_key(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> tuple[object, ...]:
    return (
        tuple(int(dim) for dim in mma_tiler_mn),
        tuple(int(dim) for dim in cluster_shape_mn),
        bool(use_2cta_instrs),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(a) for a in group_A
        ),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(b) for b in group_B
        ),
        tuple(
            _blackwell_grouped_gemm_nt_generated_tensor_cache_key(out)
            for out in out_groups
        ),
    )


def _blackwell_grouped_gemm_nt_generated_fast_call(
    bound: Callable[..., object],
) -> tuple[Callable[[], object], Any] | None:
    run = getattr(bound, "_run", None)
    run_globals = getattr(run, "__globals__", None)
    if not isinstance(run_globals, dict):
        return None

    try:
        from helion.runtime import _cute_current_stream
    except ImportError:
        return None

    for value in run_globals.values():
        entry = getattr(value, "_helion_cute_last_launch_cache", None)
        compiled = getattr(entry, "compiled", None)
        launch_args = getattr(entry, "launch_args", None)
        if callable(compiled) and isinstance(launch_args, tuple):

            def fast_call(
                compiled: Callable[..., object] = compiled,
                launch_args: tuple[object, ...] = launch_args,
            ) -> object:
                return compiled(*launch_args, _cute_current_stream())

            return fast_call, entry
    return None


def _blackwell_grouped_gemm_nt_generated_cuda_graph(
    fast_call: Callable[[], object],
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast_call()
    return graph


def _blackwell_grouped_gemm_nt_generated_call_stable_launch(
    launch: _BlackwellGeneratedStableLaunch,
) -> None:
    if (
        launch.cuda_graph is not None
        and not _blackwell_grouped_gemm_nt_generated_capture_active()
    ):
        launch.cuda_graph.replay()
        return
    if launch.fast_call is not None:
        launch.fast_call()
    else:
        launch.bound(*launch.kernel_args)


def _blackwell_grouped_gemm_nt_generated_last_stable_launch(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
) -> _BlackwellGeneratedStableLaunch | None:
    last_launch = _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH
    if last_launch is None:
        return None
    if not _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
        last_launch.guard,
        group_A,
        group_B,
        out_groups,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    ):
        return None
    return last_launch


def _blackwell_grouped_gemm_nt_generated_args(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None:
    if tuple(int(dim) for dim in mma_tiler_mn) != (128, 64):
        return None
    if tuple(int(dim) for dim in cluster_shape_mn) != (1, 1):
        return None
    if use_2cta_instrs:
        return None
    if _blackwell_grouped_gemm_nt_generated_capture_active():
        return None

    a_tuple = tuple(group_A)
    b_tuple = tuple(group_B)
    out_tuple = tuple(out_groups)
    try:
        metadata = _blackwell_grouped_gemm_nt_generated_metadata(
            a_tuple,
            b_tuple,
            out_tuple,
        )
    except ValueError:
        return None
    if metadata.a_mode0_major or metadata.b_mode0_major or metadata.c_mode0_major:
        return None
    if any(problem.k % 16 != 0 for problem in metadata.problems):
        return None

    block_m = 128
    aligned_m = tuple(
        ((problem.m + block_m - 1) // block_m) * block_m
        for problem in metadata.problems
    )
    padded_m = sum(aligned_m)
    max_n = max(problem.n for problem in metadata.problems)
    max_k = max(problem.k for problem in metadata.problems)
    if max_n % 64 != 0 or max_k % 64 != 0:
        return None
    device = metadata.device

    layout = torch.empty(padded_m, device=device, dtype=torch.int32)
    cursor = 0
    for group, (problem, group_aligned_m) in enumerate(
        zip(metadata.problems, aligned_m, strict=True)
    ):
        layout[cursor : cursor + problem.m].fill_(group)
        if problem.m != group_aligned_m:
            layout[cursor + problem.m : cursor + group_aligned_m].fill_(-1)
        cursor += group_aligned_m

    n_sizes = torch.tensor(
        [problem.n for problem in metadata.problems],
        device=device,
        dtype=torch.int32,
    )
    k_sizes = torch.tensor(
        [problem.k for problem in metadata.problems],
        device=device,
        dtype=torch.int32,
    )
    direct_pointers = torch.tensor(
        [
            (int(a.data_ptr()), int(b.data_ptr()), int(out.data_ptr()))
            for a, b, out in zip(a_tuple, b_tuple, out_tuple, strict=True)
        ],
        device=device,
        dtype=torch.int64,
    )
    direct_strides = torch.tensor(
        [
            (
                (int(a.stride(0)), int(a.stride(1))),
                (int(b.stride(0)), int(b.stride(1))),
                (int(out.stride(0)), int(out.stride(1))),
            )
            for a, b, out in zip(a_tuple, b_tuple, out_tuple, strict=True)
        ],
        device=device,
        dtype=torch.int32,
    )

    a_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (padded_m, max_k),
        dtype=a_tuple[0].dtype,
        device=device,
    )
    b_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (len(metadata.problems), max_n, max_k),
        dtype=a_tuple[0].dtype,
        device=device,
    )
    out_placeholder = _blackwell_grouped_gemm_nt_placeholder(
        (padded_m, max_n),
        dtype=out_tuple[0].dtype,
        device=device,
    )
    return (
        (
            a_placeholder,
            b_placeholder,
            layout,
            n_sizes,
            k_sizes,
            out_placeholder,
            direct_pointers,
            direct_strides,
        ),
        out_tuple,
    )


def _blackwell_grouped_gemm_nt_generated(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    out_groups: Sequence[torch.Tensor],
    *,
    mma_tiler_mn: Sequence[int],
    cluster_shape_mn: Sequence[int],
    use_2cta_instrs: bool,
    cache_stable_launch: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    global _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH

    stable_launch_key: tuple[object, ...] | None = None
    if cache_stable_launch:
        cached_launch = _blackwell_grouped_gemm_nt_generated_last_stable_launch(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
        )
        if cached_launch is not None:
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(
                cached_launch.cache_key
            )
            _blackwell_grouped_gemm_nt_generated_call_stable_launch(cached_launch)
            return tuple(out_groups)
        stable_launch_key = _blackwell_grouped_gemm_nt_generated_stable_launch_key(
            group_A,
            group_B,
            out_groups,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
        )
        cached_launch = _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.get(stable_launch_key)
        if cached_launch is not None:
            if _blackwell_grouped_gemm_nt_generated_stable_launch_guard_matches(
                cached_launch.guard,
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                use_2cta_instrs=use_2cta_instrs,
            ):
                _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(stable_launch_key)
                _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = cached_launch
                _blackwell_grouped_gemm_nt_generated_call_stable_launch(cached_launch)
                return tuple(out_groups)
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.discard(stable_launch_key)
        if _blackwell_grouped_gemm_nt_generated_capture_active():
            return None

    prepared = _blackwell_grouped_gemm_nt_generated_args(
        group_A,
        group_B,
        out_groups,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )
    if prepared is None:
        return None

    kernel_args, out_tuple = prepared
    config = _blackwell_grouped_gemm_nt_generated_config()
    block_m, block_n = _blackwell_grouped_gemm_nt_generated_config_block_mn(config)
    block_sizes = config.config["block_sizes"]
    assert isinstance(block_sizes, list)
    block_k = int(block_sizes[2])
    use_deep_ab_stages = max(int(a.size(1)) for a in group_A) >= 3 * block_k
    reserved_sms = _blackwell_grouped_gemm_nt_generated_reserved_sms(
        group_A,
        group_B,
        block_m=block_m,
        block_n=block_n,
    )
    if reserved_sms:
        config = _blackwell_grouped_gemm_nt_generated_config(reserved_sms=reserved_sms)
    if use_deep_ab_stages:
        config.config["tcgen05_ab_stages"] = 4
    bound = _blackwell_grouped_gemm_nt_generated_kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(config)
    bound(*kernel_args)
    if stable_launch_key is not None:
        stable_fast_call = _blackwell_grouped_gemm_nt_generated_fast_call(bound)
        stable_cuda_graph = (
            _blackwell_grouped_gemm_nt_generated_cuda_graph(stable_fast_call[0])
            if stable_fast_call is not None
            else None
        )
        stable_launch = _BlackwellGeneratedStableLaunch(
            cache_key=stable_launch_key,
            guard=_blackwell_grouped_gemm_nt_generated_stable_launch_guard(
                group_A,
                group_B,
                out_groups,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                use_2cta_instrs=use_2cta_instrs,
            ),
            kernel_args=kernel_args,
            bound=bound,
            fast_call=stable_fast_call[0] if stable_fast_call is not None else None,
            cuda_graph=stable_cuda_graph,
            runtime_cache_entry=(
                stable_fast_call[1] if stable_fast_call is not None else None
            ),
        )
        _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE[stable_launch_key] = stable_launch
        _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.move_to_end(stable_launch_key)
        _BLACKWELL_GENERATED_LAST_STABLE_LAUNCH = stable_launch
        while (
            len(_BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE)
            > _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE_MAX
        ):
            _BLACKWELL_GENERATED_STABLE_LAUNCH_CACHE.popitem(last=False)
    return out_tuple


def blackwell_grouped_gemm_nt_direct(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    out_groups: Sequence[torch.Tensor] | None = None,
    out_dtype: torch.dtype | None = None,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Run the explicit generated direct-pointer Blackwell grouped GEMM route.

    This opt-in path is for callers that intentionally want the generated
    direct-pointer kernel. Unsupported shapes or configs raise ``ValueError``
    instead of silently falling back to the general grouped GEMM harness.
    """
    a_tuple = tuple(group_A)
    b_tuple = tuple(group_B)
    explicit_out_groups = out_groups is not None
    if out_groups is None:
        if not a_tuple:
            raise ValueError("grouped GEMM requires at least one group")
        if len(a_tuple) != len(b_tuple):
            raise ValueError("A and B group counts must match")
        output_dtype = a_tuple[0].dtype if out_dtype is None else out_dtype
        out_tuple = tuple(
            torch.empty(
                int(a.size(0)),
                int(b.size(0)),
                device=a.device,
                dtype=output_dtype,
            )
            for a, b in zip(a_tuple, b_tuple, strict=True)
        )
    else:
        if out_dtype is not None:
            raise ValueError("out_dtype cannot be provided with explicit out_groups")
        out_tuple = tuple(out_groups)

    generated = _blackwell_grouped_gemm_nt_generated(
        a_tuple,
        b_tuple,
        out_tuple,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
        cache_stable_launch=explicit_out_groups,
    )
    if generated is None:
        raise ValueError(
            "blackwell_grouped_gemm_nt_direct does not support this shape or config"
        )
    return generated


def blackwell_grouped_gemm_nt(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    out_groups: Sequence[torch.Tensor] | None = None,
    out_dtype: torch.dtype | None = None,
    mma_tiler_mn: Sequence[int] = (128, 64),
    cluster_shape_mn: Sequence[int] = (1, 1),
    use_2cta_instrs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Run a Blackwell CuTeDSL-style grouped GEMM over independent NT problems.

    Each group computes ``A_g @ B_g.T`` where ``A_g`` has shape ``[M_g, K_g]``
    and ``B_g`` has shape ``[N_g, K_g]``. Unlike the DeepGEMM contiguous
    harness above, both ``N_g`` and ``K_g`` may differ by group.

    The CuTe backend validates the Blackwell grouped GEMM layout contract:
    A/B must share fp16 or bf16 dtype, outputs may be fp16, bf16, or fp32, all
    groups must use the same majorness for A/B/output, and each data pointer,
    contiguous dimension, and leading stride must be 16-byte aligned.
    """
    from helion._compiler.cute.grouped_deepgemm import (
        blackwell_grouped_gemm_nt as _blackwell_grouped_gemm_nt,
    )

    return _blackwell_grouped_gemm_nt(
        group_A,
        group_B,
        out_groups,
        out_dtype=out_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_2cta_instrs=use_2cta_instrs,
    )


def _blackwell_grouped_gemm_elements_per_16b(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    if dtype == torch.float32:
        return 4
    raise ValueError("dtype must be torch.float16, torch.bfloat16, or torch.float32")


def make_blackwell_grouped_gemm_nt_args(
    problem_sizes_mnkl: Sequence[
        tuple[int, int, int, int]
    ] = BLACKWELL_GROUPED_GEMM_DEFAULT_PROBLEM_SIZES,
    *,
    dtype: torch.dtype = torch.float16,
    out_dtype: torch.dtype | None = None,
    device: torch.device | str = DEVICE,
) -> tuple[
    tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    tuple[torch.Tensor, ...],
]:
    """
    Build inputs for the Blackwell grouped GEMM NT harness.

    ``problem_sizes_mnkl`` follows the NVIDIA example convention: one
    ``(M, N, K, L)`` tuple per group, with ``L == 1``. This helper creates
    row-major contiguous tensors, so each ``K`` must be a multiple of the
    A/B dtype's 16-byte element count and each ``N`` must be a multiple of the
    output dtype's 16-byte element count.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("dtype must be torch.float16 or torch.bfloat16")
    output_dtype = dtype if out_dtype is None else out_dtype
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "out_dtype must be torch.float16, torch.bfloat16, or torch.float32"
        )
    if not problem_sizes_mnkl:
        raise ValueError("problem_sizes_mnkl must contain at least one group")

    ab_elements_per_16b = _blackwell_grouped_gemm_elements_per_16b(dtype)
    out_elements_per_16b = _blackwell_grouped_gemm_elements_per_16b(output_dtype)
    group_A: list[torch.Tensor] = []
    group_B: list[torch.Tensor] = []
    for group, (m, n, k, l_mode) in enumerate(problem_sizes_mnkl):
        if l_mode != 1:
            raise ValueError("Blackwell grouped GEMM harness requires L == 1")
        if m <= 0 or n <= 0 or k <= 0:
            raise ValueError(f"group {group} problem dimensions must be positive")
        if k % ab_elements_per_16b != 0:
            raise ValueError(
                f"group {group} K must be a multiple of {ab_elements_per_16b} "
                "elements for 16-byte contiguous-dimension alignment"
            )
        if n % out_elements_per_16b != 0:
            raise ValueError(
                f"group {group} N must be a multiple of {out_elements_per_16b} "
                "elements for 16-byte contiguous-dimension alignment"
            )
        group_A.append(torch.randn(m, k, device=device, dtype=dtype).contiguous())
        group_B.append(torch.randn(n, k, device=device, dtype=dtype).contiguous())

    expected = _reference_blackwell_grouped_gemm_nt(
        group_A,
        group_B,
        out_dtype=output_dtype,
    )
    return (tuple(group_A), tuple(group_B)), expected


# %%
# TritonBench Integration Wrappers
# --------------------------------


# %%
def grouped_gemm_jagged_tritonbench(
    tb_op: object,
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor],
    w: torch.Tensor | None = None,
    split: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    """Adapter for basic grouped GEMM kernel to work with TritonBench benchmark suite."""

    def inner() -> torch.Tensor:
        A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
        return grouped_gemm_jagged(A_packed, B_shared, group_offsets)

    return inner


def grouped_gemm_jagged_persistent_tritonbench(
    tb_op: object,
    group_A: list[torch.Tensor],
    group_B: list[torch.Tensor],
    w: torch.Tensor | None = None,
    split: torch.Tensor | None = None,
) -> Callable[[], torch.Tensor]:
    """Adapter for persistent grouped GEMM kernel with dynamic work distribution for TritonBench."""

    def inner() -> torch.Tensor:
        A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
        return grouped_gemm_jagged_persistent(
            A_packed,
            B_shared,
            group_offsets,
        )

    return inner


# %%
# Reference Implementation for Validation
# ---------------------------------------


# %%
def _reference_grouped_gemm(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    B_shared = group_B[0]
    outs = [a @ B_shared for a in group_A]
    return torch.cat(outs, dim=0)


def _reference_blackwell_grouped_gemm_nt(
    group_A: Sequence[torch.Tensor],
    group_B: Sequence[torch.Tensor],
    *,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    if len(group_A) != len(group_B):
        raise ValueError("group_A and group_B must have the same number of groups")
    if not group_A:
        raise ValueError("group_A must contain at least one group")
    output_dtype = group_A[0].dtype if out_dtype is None else out_dtype
    return tuple(
        (a.float() @ b.float().T).to(output_dtype)
        for a, b in zip(group_A, group_B, strict=True)
    )


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


def grouped_gemm_jagged_example(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    """
    Wrapper to run grouped_gemm_jagged with unpacked TritonBench inputs.
    """
    A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
    return grouped_gemm_jagged(A_packed, B_shared, group_offsets)


def grouped_gemm_jagged_persistent_example(
    group_A: list[torch.Tensor], group_B: list[torch.Tensor]
) -> torch.Tensor:
    """
    Wrapper to run grouped_gemm_jagged_persistent with unpacked TritonBench inputs.
    """
    A_packed, B_shared, group_offsets = _pack_group_inputs(group_A, group_B)
    return grouped_gemm_jagged_persistent(A_packed, B_shared, group_offsets)


# %%
# Test Harness and Validation
# ---------------------------


# %%
def main() -> None:
    torch.manual_seed(0)  # Ensure reproducible test results
    device = DEVICE
    dtype = torch.bfloat16
    G = 4  # Number of groups to test
    K, N = 256, 128  # Shared dimensions: K (reduction), N (output columns)
    # Create test data with varying group sizes (M_i = 64, 128, 192, 256)
    group_A = [
        torch.randn(64 * (i + 1), K, device=device, dtype=dtype).contiguous()
        for i in range(G)
    ]
    # Shared weight matrix B replicated for each group (as per TritonBench convention)
    group_B = [torch.randn(K, N, device=device, dtype=dtype).contiguous()] * G

    print("Testing grouped GEMM kernels...")
    run_example(
        grouped_gemm_jagged_example,
        _reference_grouped_gemm,
        (group_A, group_B),
        rtol=1e-2,
        atol=1e-2,
    )
    print("✓ Non-persistent kernel passed")

    run_example(
        grouped_gemm_jagged_persistent_example,
        _reference_grouped_gemm,
        (group_A, group_B),
        rtol=1e-2,
        atol=1e-2,
    )
    print("✓ Persistent kernel passed")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
