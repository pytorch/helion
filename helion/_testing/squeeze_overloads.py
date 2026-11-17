"""Shared squeeze-overload kernel for source and native tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def squeeze_overloads(x: torch.Tensor, mode: int) -> torch.Tensor:
    mode = hl.specialize(mode)
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        value = x[tile]
        if mode == 0:
            value = value.squeeze()
        else:
            value = value.unsqueeze(0).unsqueeze(-1)
            if mode == 1:
                value = value.squeeze()
            elif mode == 2:
                value = value.squeeze((0, -1))
            else:
                # Requested non-singleton dimensions remain intact.
                value = value.squeeze((0, 1, -1))
        out[tile] = value.sin()
    return out
