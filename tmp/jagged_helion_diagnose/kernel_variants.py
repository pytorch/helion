"""Same kernel body as examples/jagged_sum_tpu.py, three times — once per
``pallas_loop_type`` value. Used by diagnose.py to isolate whether the
all-zeros bug is emit_pipeline-specific or affects every Pallas codegen
path.

The kernel body is intentionally duplicated rather than factored: each
``@helion.kernel`` decorator must wrap a separate function so the
compiler treats them as distinct kernels (and so we can pin a different
``pallas_loop_type`` on each).
"""
from __future__ import annotations

import torch

import helion
import helion.language as hl


_BASE_CFG = dict(block_sizes=[32, 128], pallas_pre_broadcast=False)


@helion.kernel(
    config=helion.Config(pallas_loop_type="emit_pipeline", **_BASE_CFG),
    static_shapes=True,
)
def kernel_emit_pipeline(
    x_data: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    M = x_data.shape[1]
    N = x_offsets.shape[0] - 1
    L = x_data.shape[0]
    out = torch.zeros([N, M], dtype=x_data.dtype, device=x_data.device)
    for tile_l in hl.tile(L):
        global_row = tile_l.index
        
        for tile_m in hl.tile(M):
            chunk = x_data[tile_l, tile_m]
            for item_idx in range(N):
                it_start = x_offsets[item_idx]
                it_end = x_offsets[item_idx + 1]
                
                mask = (global_row >= it_start) & (global_row < it_end)
                safe_chunk = (mask * 1.0)[:, None] * chunk
                partial = safe_chunk.sum(dim=0)
                hl.atomic_add(out, [item_idx, tile_m], partial)
    return out


@helion.kernel(
    config=helion.Config(pallas_loop_type="fori_loop", **_BASE_CFG),
    static_shapes=True,
)
def kernel_fori_loop(
    x_data: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    M = x_data.shape[1]
    N = x_offsets.shape[0] - 1
    L = x_data.shape[0]
    out = torch.zeros([N, M], dtype=x_data.dtype, device=x_data.device)
    for tile_l in hl.tile(L):
        global_row = tile_l.index
        
        for tile_m in hl.tile(M):
            chunk = x_data[tile_l, tile_m]
            for item_idx in range(N):
                it_start = x_offsets[item_idx]
                it_end = x_offsets[item_idx + 1]
                
                mask = (global_row >= it_start) & (global_row < it_end)
                safe_chunk = (mask * 1.0)[:, None] * chunk
                partial = safe_chunk.sum(dim=0)
                hl.atomic_add(out, [item_idx, tile_m], partial)
    return out


@helion.kernel(
    config=helion.Config(pallas_loop_type="unroll", **_BASE_CFG),
    static_shapes=True,
)
def kernel_unroll(
    x_data: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    M = x_data.shape[1]
    N = x_offsets.shape[0] - 1
    L = x_data.shape[0]
    out = torch.zeros([N, M], dtype=x_data.dtype, device=x_data.device)
    for tile_l in hl.tile(L):
        global_row = tile_l.index
        
        for tile_m in hl.tile(M):
            chunk = x_data[tile_l, tile_m]
            for item_idx in range(N):
                it_start = x_offsets[item_idx]
                it_end = x_offsets[item_idx + 1]
                
                mask = (global_row >= it_start) & (global_row < it_end)
                safe_chunk = (mask * 1.0)[:, None] * chunk
                partial = safe_chunk.sum(dim=0)
                hl.atomic_add(out, [item_idx, tile_m], partial)
    return out


VARIANTS = {
    "emit_pipeline": kernel_emit_pipeline,
    "fori_loop": kernel_fori_loop,
    "unroll": kernel_unroll,
}
