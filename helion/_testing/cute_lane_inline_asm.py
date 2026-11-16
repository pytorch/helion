"""One opaque lane producer with visible memory and multiplicity effects."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def snapshot_before_opaque_producer(
    x: torch.Tensor,
    overwrite: torch.Tensor,
    counters: torch.Tensor,
    shared: hl.constexpr,
) -> torch.Tensor:
    if shared:
        out = torch.empty_like(x)
    else:
        out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        values = x[row, :]
        destinations = overwrite[row, :]
        count = counters[row, :]
        preserved = hl.inline_asm_elementwise(
            "{ .reg .b32 old; atom.global.add.u32 old, [$2], 1; "
            "st.global.u32 [$1], 0; mov.f32 $0, $3; }",
            "=f,l,l,f",
            [destinations, count, values],
            torch.float32,
            False,
            1,
        )
        if shared:
            out[row, :] = preserved - preserved.sum(-1, keepdim=True)
        else:
            out[row] = preserved.sum(-1)
    return out
