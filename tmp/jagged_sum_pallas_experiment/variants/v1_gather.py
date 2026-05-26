"""V1 baseline — copy of examples/jagged_sum.py, gather-based load via hl.load(x_flat, [flat_indices]).

Purpose: establish what Helion's current Pallas backend does with the existing
source. Expected outcome: either (a) compiles to a gather Pallas can't run on
TPU, (b) compiles to something with masked emit_pipeline, or (c) hard error.
The printed lowered code tells us which.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running this script directly from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from _common import run_all_sizes  # noqa: E402


@helion.kernel(autotune_effort="none")
def jagged_sum_gather(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros(
        [num_rows, M], dtype=x_data.dtype, device=x_data.device
    )
    x_flat = x_data.view(-1)

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts

        for tile_m in hl.tile(M):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
            for tile_k in hl.jagged_tile(nnz):
                base = starts[:, None] + tile_k.index[None, :]
                flat = base[:, :, None] * M + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat])
                row_sums = row_sums + x_slice.sum(dim=1)
            out[tile_b, tile_m] = row_sums
    return out


def main() -> int:
    return run_all_sizes(
        "V1 — gather-based source (current examples/jagged_sum.py pattern)",
        jagged_sum_gather,
    )


if __name__ == "__main__":
    sys.exit(main())
