"""The Metal benchmark's Helion kernel for grouped_gemm (see grouped_gemm.py)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def grouped_gemm_jagged_grid(
    A_packed: torch.Tensor,  # [total_M, K]
    B: torch.Tensor,  # [K, N]
    group_offsets: torch.Tensor,  # [G+1], row offsets into A_packed
) -> torch.Tensor:
    """Grouped GEMM over one grid of (group, M tile, N tile) work items.

    Every group gets as many M tiles as all groups' rows together could need,
    so the grid's size is known on the host; a work item past its group's
    rows does nothing (about G times the real tiles for equal groups, G = 4
    here).  The config's ``pid_type`` runs the grid flat or as persistent
    workers, and ``num_sm_multiplier`` sizes the workers.
    """
    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2

    # Every tile of every group is some work item, so the kernel writes every
    # row and the output needs no zero-fill.
    out = torch.empty(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1
    num_m_tiles = (total_M + BLOCK_M - 1) // BLOCK_M
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N

    for g, m_tile, n_tile in hl.grid([G, num_m_tiles, num_n_tiles]):
        group_start = group_offsets[g]
        group_end = group_offsets[g + 1]
        base_row = group_start + m_tile * BLOCK_M
        if base_row < group_end:
            row_idx = base_row + hl.arange(BLOCK_M)
            # pyrefly: ignore[unsupported-operation]
            col_idx = n_tile * BLOCK_N + hl.arange(BLOCK_N)
            rows_valid = row_idx < group_end
            cols_valid = col_idx < N

            acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)
            for k_tile in hl.tile(K):
                k_idx = k_tile.index
                a_blk = hl.load(
                    A_packed,
                    [row_idx, k_idx],
                    extra_mask=rows_valid[:, None],  # pyrefly: ignore[bad-index]
                )
                b_blk = hl.load(
                    B,
                    [k_idx, col_idx],
                    extra_mask=cols_valid[None, :],  # pyrefly: ignore[bad-index]
                )
                acc = torch.addmm(acc, a_blk, b_blk)

            # pyrefly: ignore[bad-index]
            valid_2d = rows_valid[:, None] & cols_valid[None, :]
            hl.store(out, [row_idx, col_idx], acc.to(out.dtype), extra_mask=valid_2d)

    return out
