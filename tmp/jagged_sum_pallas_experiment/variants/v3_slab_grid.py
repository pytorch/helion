"""V3 — hl.grid outer, direct slab access x_data[start:end, :] inside.

The most "Pallas-natural" expression of jagged_sum: one program per batch,
slice the input slab, reduce. If Helion accepts dynamic slicing of this form,
it's the closest source pattern to a BlockSpec+BoundedSlice lowering.

Likely failure modes:
  - "Dynamic slicing not supported" — Helion may reject `x_data[start:end, :]`
    when start/end are loaded scalars (most likely outcome).
  - Triton-style fallback to per-element load (not what we want).
  - Compiles but emits gather under the hood — visible in printed code.
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
def jagged_sum_slab(
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
        # Dynamic slab access: rows [start:end), all cols.
        x_slab = x_data[start:end, :]
        out[i, :] = x_slab.sum(dim=0)
    return out


def main() -> int:
    banner("V3 — hl.grid + direct slab slice x_data[start:end, :]")
    x_data, x_offsets = make_data()
    report_inputs(x_data, x_offsets)

    try:
        out = jagged_sum_slab(x_data, x_offsets)
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
