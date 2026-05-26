"""V4 — hl.grid outer + hl.jagged_tile inner. Hybrid: per-batch program,
jagged-tile over the inner length.

Reason for trying: v3's `x_data[start:end, :]` may be rejected by Helion as
"dynamic slicing not supported." Falling back to one-program-per-batch +
jagged_tile lets us keep the per-batch shape known scalar at the outer level,
while still using a Helion-supported tiled inner load.

The inner access `x_data[start + tile_k.index, :]` is still a fancy index,
but with a SCALAR start (vs. v2's [tile_b]-vector start). That scalar form
should map naturally to a strided DMA: base address + tile_k.block_size rows.
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
def jagged_sum_grid_jagged(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    M = x_data.shape[1]
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros(
        [num_rows, M], dtype=x_data.dtype, device=x_data.device
    )

    for i in hl.grid(num_rows):
        start = x_offsets[i]
        end = x_offsets[i + 1]
        nnz = end - start
        acc = hl.zeros([M], dtype=x_data.dtype)
        for tile_k in hl.jagged_tile(nnz):
            idx = start + tile_k.index  # 1D scalar-start index
            x_block = x_data[idx, :]    # [tile_k, M] — fancy index axis 0, full axis 1
            acc = acc + x_block.sum(dim=0)
        out[i, :] = acc
    return out


def main() -> int:
    banner("V4 — hl.grid (outer per-batch) + hl.jagged_tile (inner)")
    x_data, x_offsets = make_data()
    report_inputs(x_data, x_offsets)

    try:
        out = jagged_sum_grid_jagged(x_data, x_offsets)
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
