"""Generic pointwise comparisons at signed Int32 scalar boundaries."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", autotune_effort="none")
def integer_literal_comparisons(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty((8, x.numel()), dtype=torch.bool, device=x.device)
    for tile in hl.tile(x.numel()):
        value = x[tile]
        out[0, tile] = value < 2**31
        out[1, tile] = value >= -(2**31) - 1
        out[2, tile] = value == 2**32
        out[3, tile] = value != -(2**32) - 1
        out[4, tile] = value <= 2**31 - 1
        out[5, tile] = value < -(2**31)
        out[6, tile] = value > -(2**32)
        out[7, tile] = value <= 2**32 + 1
    return out


def integer_literal_inputs(device: torch.device) -> torch.Tensor:
    values = torch.tensor(
        [-(2**31), -(2**31) + 1, -1, 0, 1, 2**31 - 2, 2**31 - 1],
        dtype=torch.int32,
        device=device,
    )
    return values.repeat(6)[:37]


def integer_literal_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x < 2**31,
            x >= -(2**31) - 1,
            x == 2**32,
            x != -(2**32) - 1,
            x <= 2**31 - 1,
            x < -(2**31),
            x > -(2**32),
            x <= 2**32 + 1,
        ]
    )
