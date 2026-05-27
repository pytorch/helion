"""Minimal repro of the structural pattern that produced all-zero output
in the three loop_type variants of jagged_sum_tpu.

Same iteration shape and same `out[i, tile_m] += partial` access pattern,
but `partial` is a deterministic constant (1.0 per cell) — no mask, no
offsets, no jagged logic. The only thing being exercised is:

    outer tile_l index is NOT in the output index_map -> all L-programs
    write to the same output rows; the += chain has to chain through
    the BlockSpec for `out`.

If the result is the expected non-zero accumulation (all cells = L // block_L),
the structural pattern is fine and the bug is somewhere in the mask /
sum / fp32 / partial-value computation of the original kernel.

If the result is all zeros, the bug is purely structural — Helion's
Pallas codegen silently drops in-place accumulator writes to outputs
whose index isn't parameterized by the outer tile axis.
"""
from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[32, 128],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=False,
    ),
    static_shapes=True,
)
def repro_kernel(
    x_data: torch.Tensor,        # [L, M] — only used to keep tile_l live
    out_size_hint: torch.Tensor, # [out_size]    — out_size from shape
) -> torch.Tensor:
    out_size = out_size_hint.shape[0]
    M = x_data.shape[1]
    L = x_data.shape[0]
    out = torch.zeros(
        [out_size, M], dtype=x_data.dtype, device=x_data.device
    )
    for tile_l in hl.tile(L):
        for tile_m in hl.tile(M):
            chunk = x_data[tile_l, tile_m]
            # Constant partial of value 1.0 per column. Done via `chunk * 0.0`
            # so that `chunk` (and therefore tile_l) stay live in the IR.
            partial = (chunk * 0.0).sum(dim=0) + 1.0  # (block_M,) of 1.0
            for i in range(out_size):
                out[i, tile_m] += partial
    return out
