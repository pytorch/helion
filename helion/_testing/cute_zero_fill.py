"""The atomic zero-fill fixture shared by source and native tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def atomic_column_sum(x: torch.Tensor) -> torch.Tensor:
    """Column sums through atomics into a zero-filled fp32 accumulator: the
    fill cannot be elided (no CTA writes a whole column)."""
    m, n = x.size()
    out = torch.zeros([n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        hl.atomic_add(out, [tile_n], x[tile_m, tile_n].to(torch.float32).sum(0))
    return out


COLUMN_SUM_CONFIG = {"block_sizes": [64, 64], "num_threads": [4, 64]}
