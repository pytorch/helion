from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    for tile in hl.tile(output.size()):
        output[tile] = x[tile] + y[tile]

    return output
