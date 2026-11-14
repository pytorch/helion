from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(autotune_effort="none")
def rebound_host_float(
    x: torch.Tensor, alpha: float, branch: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_b in hl.tile(x.size(0)):
        for tile_m in hl.tile(x.size(1)):
            value = alpha + 0.0
            if branch:
                if x[0, 0, 0] > 0:
                    value = value * 2.0
            else:
                for _tile_n in hl.tile(x.size(2)):
                    value = value * 2.0
            for tile_out in hl.tile(x.size(2)):
                out[tile_b, tile_m, tile_out] = x[tile_b, tile_m, tile_out] * value
    return out


@helion.kernel(autotune_effort="none")
def aliased_host_float(x: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_b in hl.tile(x.size(0)):
        value = alpha + 0.0
        original = value
        for _tile_m in hl.tile(x.size(1)):
            value = value * 2.0
        for tile_out in hl.tile(x.size(1)):
            out[tile_b, tile_out] = x[tile_b, tile_out] * (value + original)
    return out
