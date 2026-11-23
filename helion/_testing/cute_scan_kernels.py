"""Scan kernels shared by CuTe correctness and configuration tests."""

from __future__ import annotations

from typing import cast

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def computed_scan(
    x: torch.Tensor, axis: hl.constexpr, reverse: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    reused = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        values = x[row, :, :] + 1
        out[row, :, :] = hl.cumsum(
            values,
            dim=cast("int", axis),
            reverse=cast("bool", reverse),
            dtype=None if x.dtype.is_floating_point else x.dtype,
        )
        reused[row, :, :] = values * 2
    return out, reused
