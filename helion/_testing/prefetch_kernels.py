"""Kernels shared by the prefetch tests (test_prefetch, test_prefetch_hints).

Each test module wraps them in its own ``helion.kernel``, so no binding state
is shared between modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import helion.language as hl

if TYPE_CHECKING:
    import torch


def hint_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """A prefetch-only root reads ``ids``, which a later root rewrites."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids
