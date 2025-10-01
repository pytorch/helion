#!/usr/bin/env python3
"""Minimal reproduction for Helion symbolic size mismatch in layer-norm backward."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel
def _sum_feature_mismatch(grad_out: torch.Tensor) -> torch.Tensor:
    m, n = grad_out.shape  # grad_out: [m, n]
    n = hl.specialize(n)  # n: int (feature size)

    grad_block = torch.zeros(n, dtype=torch.float32, device=grad_out.device)  # grad_block: [n]

    for tile_m in hl.tile(m):  # tile_m: tile descriptor over batch dim
        dy_m = grad_out[tile_m, :].to(torch.float32)  # dy_m: [tile_m, n]
        grad_block[:] += torch.sum(dy_m, dim=0)  # torch.sum(..., dim=0): [n] (symbolic)

    return grad_block  # [n]


def main() -> None:
    m, n = 4096, 5632
    device = "cuda"
    dtype = torch.float16

    grad_out = torch.randn((m, n), device=device, dtype=dtype)

    print(f"Running minimal mismatch repro with shape {(m, n)}…")
    _sum_feature_mismatch(grad_out)


if __name__ == "__main__":
    main()
