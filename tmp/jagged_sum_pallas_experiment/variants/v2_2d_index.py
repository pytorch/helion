"""V2 — same hl.jagged_tile structure, but load via 2D index `x_data[base_idx, :]`
instead of flat `hl.load(x_flat, [flat_indices])`.

Hypothesis: Helion may lower the 2D-index form to a different access pattern
on Pallas — possibly something closer to a per-program slab DMA — vs. the
explicit flat gather. The shape of `base_idx` is [tile_b, tile_k] so the
resulting load is [tile_b, tile_k, M]. If Helion's Pallas backend lifts the
M dim out (since it's a full take), the load could become a per-row slab.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from _common import banner, check_close, make_data, reference, report_inputs  # noqa: E402


@helion.kernel(autotune_effort="none")
def jagged_sum_2d_index(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros(
        [num_rows, M], dtype=x_data.dtype, device=x_data.device
    )

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts

        row_sums = hl.zeros([tile_b, M], dtype=x_data.dtype)
        for tile_k in hl.jagged_tile(nnz):
            base = starts[:, None] + tile_k.index[None, :]  # [tile_b, tile_k]
            # 2D index over axis 0, full slice over axis 1.
            x_slice = x_data[base, :]  # expected shape [tile_b, tile_k, M]
            row_sums = row_sums + x_slice.sum(dim=1)

        out[tile_b, :] = row_sums
    return out


def main() -> int:
    banner("V2 — 2D-index form: x_data[base, :] inside hl.jagged_tile")
    x_data, x_offsets = make_data()
    report_inputs(x_data, x_offsets)

    try:
        out = jagged_sum_2d_index(x_data, x_offsets)
    except Exception:
        print("COMPILE/RUN FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 1

    ref = reference(x_data, x_offsets)
    try:
        check_close(out, ref)
    except AssertionError:
        print("CORRECTNESS FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
