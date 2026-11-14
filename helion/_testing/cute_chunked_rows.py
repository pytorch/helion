"""A chunked (looped) row reduction whose two sweeps walk reduction lanes.

``row_shift`` is the kernel of the CuTe reduction lane-unroll codegen tests
(``test/test_cute_reduction_lane_unroll.py``) and of their GPU twin
(``test/test_cute_device_loop_lane_unroll.py``): its reduce sweep sums a row
of ``x``, and its consume sweep loads the row of ``y``, subtracts the sum and
stores the result, so ``cute_lane_unroll`` moves the loads of both sweeps.
"""

from __future__ import annotations

import torch

import helion.language as hl


def row_shift(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows, n = y.shape
    out = torch.empty_like(y)
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        out[tile_r, :] = (y[tile_r, :].to(torch.float32) - total).to(out.dtype)
    return out
